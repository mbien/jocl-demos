
typedef struct {
	float x, y, z; // position, also color (r,g,b)
} Vec;

typedef struct {
	/* User defined values */
	Vec orig, target;
	/* Calculated values */
	Vec dir, x, y;
} Camera;

typedef struct {
	unsigned int width, height;
	int superSamplingSize;
	int actvateFastRendering;
	int enableShadow;

	unsigned int maxIterations;
	float epsilon;
	float mu[4];
	float light[3];
	Camera camera;
} RenderingConfig;
