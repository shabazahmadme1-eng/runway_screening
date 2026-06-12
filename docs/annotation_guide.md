# Annotation guide

Distilled from the client (Sequetrics) doubts-and-concerns Q&A. Use this
during the ground-truth refinement pass (CVAT / Roboflow) when reviewing
pseudo-labels and adding manual boxes.

## Ground rules

- **One tag per area.** The client confirmed multi-tagging the same region
  (e.g. "discoloration" + "rough surface") is not wanted — pick the single
  best-fitting class below.
- **Do not extrapolate.** When a marking is only partially visible, annotate
  what is actually visible; do not assume/extend the missing part.

## When to use each class

| id | class                 | use for |
|----|-----------------------|---------|
| 0  | crack                 | longitudinal / transverse / alligator cracks in the surface |
| 1  | spalling              | surface break-up, potholes, missing chunks of pavement |
| 2  | fod                   | foreign object debris: loose objects sitting on the surface |
| 3  | faded_paint_marking   | partially visible / worn runway markings, incl. pre-threshold lines (annotate the visible part only) |
| 4  | band_joint            | the joint lines between adjacent pavement bands/sections |
| 5  | gap_vegetation        | vegetation growing in the gaps between pavement bands or tiles |
| 6  | aged_surface          | older, rougher pavement areas (visible holes / coarse texture) adjacent to newer smoother surface |
| 7  | repair_patch          | patched / resurfaced repair areas |
| 8  | weathered_surface     | black tar surface turned whitish / discoloured by weathering (single tag — do not also tag as rough/aged) |
| 9  | surface_discoloration | greenish staining of the surface caused by vegetation |
| 10 | paint_marking         | scattered white paint spots / spillage (not part of an official marking) |
| 11 | faded_surface_marking | faded operational surface markings such as the closed-runway cross (X) |

## Disambiguation notes

The schema has three paint-related and three condition-related classes that
are easy to mix up:

- **faded_paint_marking vs paint_marking vs faded_surface_marking** —
  `faded_paint_marking` is a *line marking* that has worn away (threshold /
  centre / edge lines); `paint_marking` is *accidental* paint (spots,
  spillage); `faded_surface_marking` is a faded *symbol* (e.g. the X cross).
- **aged_surface vs weathered_surface vs surface_discoloration** —
  `aged_surface` is about *texture* (rough, holes); `weathered_surface` is
  about *colour change from weathering* (black → whitish);
  `surface_discoloration` is specifically the *greenish* vegetation stain.
- **gap_vegetation vs surface_discoloration** — actual plant growth in a
  joint/gap is `gap_vegetation`; a green stain on the surface itself is
  `surface_discoloration`.
