Segmented image of the Urban-6 hyperspectral dataset.
Segmentation performed using the algorithm in [B].

Four classes are represented:
* "Asphalt" - (255,   0,   0) [Red]
* "Grass"   - (  0, 255,   0) [Green]
* "Tree"    - (  0,   0, 255) [Blue]
* "Roof"    - (255, 255,   0) [Yellow]
* "Metal"   - (255,   0, 255) [Magenta]
* "Dirt"    - (  0, 255, 255) [Cyan]

PNG image of dtype uint8 and shape (100,100,3), where each RGB pixel represents one of the four classes above by its RGB vector.

[B] Zhang, Y., Wang, X., Jiang, X., Zhang, L., Du, B.: Elastic graph fusion subspace clustering for large hyperspectral image. IEEE Transactions on Circuits and Systems for Video Technology, 2025.
