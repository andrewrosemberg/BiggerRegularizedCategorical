# Wrist-Ray Sensor Setting Choice

This note records the fixed raycast settings used for the wrist-mounted
pointcloud conditioner. The camera/site pose is fixed on `robot0:palm`; the
choice here concerns only the ray grid and field of view.

## Selected Setting

Use the following default in `RaycastConfig`:

- grid: `32x32`, giving 1024 rays;
- vertical FOV: `40 deg`;
- max distance: `0.34 m`;
- ray pattern: uniform pinhole grid in the wrist-camera frame.

Use `64x64`, 40 deg only for high-quality offline data collection or diagnostic
visualization when runtime is not the bottleneck.

The setting is realistic for a close-range wrist depth sensor: a 40 deg vertical
FOV is comparable to narrow RGB-D/depth cameras used for manipulation, and 1024
points is in the common pointcloud policy input range.

## Why Not The Initial Smoke Setting

The initial smoke-test setting, `16x16` with 60 deg FOV, was useful for checking
the plumbing but too sparse for learning. Across 13 train-split objects and five
random initializations per object:

| Setting | Rays | Mean hits | Hit fraction | Mean object hits | Object-ray fraction | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `16x16`, 60 deg | 256 | 59.2 | 0.231 | 28.2 | 0.110 | 1.7 ms |
| `32x32`, 40 deg | 1024 | 462.7 | 0.452 | 220.4 | 0.215 | 7.4 ms |
| `64x64`, 40 deg | 4096 | 1863.5 | 0.455 | 882.4 | 0.215 | 28.3 ms |

The selected default gives roughly 7.8x more valid hits and 7.8x more object hits
than the smoke setting, while remaining fast enough for online use.

## Visual Evidence

Comparison figures were generated for five train-split objects. Each figure shows
the scene, wrist-camera RGB view, raycast pointcloud, and a diagnostic-only
object-vs-hand coloring. The diagnostic object-vs-hand split uses simulator geom
ids and must not be used as a policy input.

The RGB column uses the selected 40 deg debug camera FOV. It therefore matches
the selected `32x32`, 40 deg and high-quality `64x64`, 40 deg raycast rows; the
`16x16`, 60 deg row is included only as the sparse baseline comparison.

- [mug](raycast_sensor_sweep/figures/mug_setting_comparison.png)
- [hammer](raycast_sensor_sweep/figures/hammer_setting_comparison.png)
- [knife](raycast_sensor_sweep/figures/knife_setting_comparison.png)
- [cracker_box](raycast_sensor_sweep/figures/cracker_box_setting_comparison.png)
- [wine_glass](raycast_sensor_sweep/figures/wine_glass_setting_comparison.png)

The `32x32`, 40 deg setting makes object geometry recognizable in cases where the
initial `16x16`, 60 deg pointcloud is too sparse, especially for `mug`, `hammer`,
and `knife`.

## Implementation Notes

- The MuJoCo camera `fovy` is also set to 40 deg so debug RGB renders match the
  raycast frustum.
- The camera pose is not changed by this setting choice.
- `max_dist=0.34` is retained; increasing it did not improve coverage for the
  hand-object workspace.
- The `64x64`, 40 deg setting is retained as an offline/high-quality option but
  is not the default because it costs about 28 ms per raycast with the current
  single-threaded `mj_ray` loop.
