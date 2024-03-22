Implementation Details:

Padding type: 	Zero Padding
Padding:		4 pixels each side

Training:		Random horizontal flipping, cropping 32x32 from images
Evaluation:		Original Image

Optimizer:		SGD
Momentum:		0.9
Batch size:		128
Weight Decay:	5e-4
Learning Rate:	0.1 (Updated after 32,000 and 48,000 steps by division by 10)

SimAM Lambda:	1e-4 (Searched)
