# Failure-Case Analysis

This document will be expanded throughout the project.

## Initial expected failure modes

- Low contrast between object and background
- Strong shadows or highlights
- Motion blur or defocus blur
- Partial occlusion
- Cluttered scenes with touching objects
- Symmetric objects where orientation is visually ambiguous
- Detection failure on object categories not represented in training or examples

## How we will analyze failures

For each failure case, we will document:

- input image
- expected behavior
- actual behavior
- suspected cause
- possible fix
- whether the issue is acceptable for a demo portfolio project
