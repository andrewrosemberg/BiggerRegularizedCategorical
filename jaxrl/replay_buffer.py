import gymnasium as gym
import numpy as np

import os
import pickle

from jaxrl.utils import Batch

class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int):
        self.observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((num_tasks, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4
        self.num_tasks = num_tasks

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size * self.num_tasks, size=(num_batches, batch_size))
        task_indx, sample_indx = np.divmod(indx, self.size)
        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations,
                     task_ids=task_indx)    

    def sample_task_batches(self):
        batch_size = 32
        indxs = np.random.randint(self.size, size=batch_size)        
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids)  
        
    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))


class ObjectAwareReplayBuffer:
    """Replay buffer indexed by object ID, not environment slot.

    Multiple env slots map to the same object. Each object has its own
    circular write pointer and valid-entry count. Insert is fully
    vectorized across slots and objects.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_objects: int, slot_to_object: np.ndarray):
        obs_dim = observation_space.shape[-1]
        obs_dtype = observation_space.dtype
        self.observations = np.empty((num_objects, capacity, obs_dim), dtype=obs_dtype)
        self.actions = np.empty((num_objects, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_objects, capacity), dtype=np.float32)
        self.masks = np.empty((num_objects, capacity), dtype=np.float32)
        self.next_observations = np.empty((num_objects, capacity, obs_dim), dtype=obs_dtype)

        self.capacity = capacity
        self.num_tasks = num_objects
        self.num_objects = num_objects

        self._slot_to_object = np.asarray(slot_to_object, dtype=np.int64)
        if self._slot_to_object.ndim != 1:
            raise ValueError("slot_to_object must be a one-dimensional array.")
        if self._slot_to_object.size == 0:
            raise ValueError("slot_to_object must contain at least one slot.")
        if self._slot_to_object.min() < 0 or self._slot_to_object.max() >= num_objects:
            raise ValueError(
                "slot_to_object entries must be object IDs in "
                f"[0, {num_objects})."
            )
        num_slots = len(self._slot_to_object)

        slots_per_obj = np.bincount(self._slot_to_object, minlength=num_objects)
        if np.any(slots_per_obj == 0):
            empty = np.where(slots_per_obj == 0)[0]
            raise ValueError(
                "Every object must have at least one env slot; "
                f"empty objects: {empty}."
            )
        if np.max(slots_per_obj) > capacity:
            raise ValueError(
                "ObjectAwareReplayBuffer capacity must be at least the maximum "
                f"number of env slots per object; got capacity={capacity}, "
                f"max_slots_per_object={int(np.max(slots_per_obj))}."
            )
        self._slots_per_object = slots_per_obj.astype(np.int64)

        self._slot_offset = np.empty(num_slots, dtype=np.int64)
        _counter = np.zeros(num_objects, dtype=np.int64)
        for s in range(num_slots):
            obj = self._slot_to_object[s]
            self._slot_offset[s] = _counter[obj]
            _counter[obj] += 1

        self._insert_index = np.zeros(num_objects, dtype=np.int64)
        self._size = np.zeros(num_objects, dtype=np.int64)

    @property
    def size(self):
        return int(self._size.min())

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: np.ndarray, mask: np.ndarray, next_observation: np.ndarray):
        obj = self._slot_to_object
        targets = (self._insert_index[obj] + self._slot_offset) % self.capacity
        self.observations[obj, targets] = observation
        self.actions[obj, targets] = action
        self.rewards[obj, targets] = reward
        self.masks[obj, targets] = mask
        self.next_observations[obj, targets] = next_observation
        self._insert_index = (self._insert_index + self._slots_per_object) % self.capacity
        self._size = np.minimum(self._size + self._slots_per_object, self.capacity)

    def sample(self, batch_size: int, num_batches: int):
        valid = np.where(self._size > 0)[0]
        if valid.size == 0:
            raise ValueError("Cannot sample from an empty ObjectAwareReplayBuffer.")
        obj_ids = np.random.choice(valid, size=(num_batches, batch_size))
        sizes = self._size[obj_ids]
        sample_idx = (np.random.random((num_batches, batch_size)) * sizes).astype(np.int64)
        return Batch(
            observations=self.observations[obj_ids, sample_idx],
            actions=self.actions[obj_ids, sample_idx],
            rewards=self.rewards[obj_ids, sample_idx],
            masks=self.masks[obj_ids, sample_idx],
            next_observations=self.next_observations[obj_ids, sample_idx],
            task_ids=obj_ids,
        )

    def sample_task_batches(self):
        if np.any(self._size == 0):
            empty = np.where(self._size == 0)[0]
            raise ValueError(
                "Cannot sample task batches before every object has data; "
                f"empty objects: {empty}."
            )
        batch_size = 32
        arange_obj = np.arange(self.num_objects)
        obj_ids = arange_obj[:, None] + np.zeros(batch_size, dtype=np.int32)
        sizes = self._size[:, None]
        idx = (np.random.random((self.num_objects, batch_size)) * sizes).astype(np.int64)
        return Batch(
            observations=self.observations[arange_obj[:, None], idx],
            actions=self.actions[arange_obj[:, None], idx],
            rewards=self.rewards[arange_obj[:, None], idx],
            masks=self.masks[arange_obj[:, None], idx],
            next_observations=self.next_observations[arange_obj[:, None], idx],
            task_ids=obj_ids,
        )
