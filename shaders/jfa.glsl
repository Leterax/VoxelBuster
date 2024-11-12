#version 450

// Define the chunk size
const uint chunk_size = 16u;

// Binding points for the seed position images (ping-pong buffers)
layout(binding = 0, rgba32f) uniform image3D positions0;
layout(binding = 1, rgba32f) uniform image3D positions1;

// Binding point for the final distance field output
layout(binding = 2, r32f) uniform image3D distanceField;

// Local workgroup size (adjust as needed)
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

// Function prototype for getVoxel (to be implemented elsewhere)
bool getVoxel(uvec3 pos);

// Main compute shader entry point
void main() {
    uvec3 p = gl_GlobalInvocationID;
    if (any(greaterThanEqual(p, uvec3(chunk_size)))) {
        return; // Out of bounds
    }

    // Initialization Phase
    // --------------------
    // Initialize the seed positions
    if (getVoxel(p)) {
        // Occupied voxel (seed)
        imageStore(positions0, ivec3(p), vec4(vec3(p), 1.0)); // w = 1.0 indicates seed assigned
    } else {
        // Empty voxel
        imageStore(positions0, ivec3(p), vec4(0.0, 0.0, 0.0, 0.0)); // w = 0.0 indicates no seed
    }

    // Memory barrier to ensure all writes are completed
    barrier();
    memoryBarrierImage();

    // Jump Flood Algorithm Iterations
    // -------------------------------
    // Define the step sizes
    int ks[4] = int[](8, 4, 2, 1);
    // Ping-pong buffers indices
    bool toggle = false;

    for (int s = 0; s < 4; s++) {
        int k = ks[s];

        // Read from the current buffer
        vec4 pSeed = imageLoad(toggle ? positions1 : positions0, ivec3(p));

        // Iterate over neighbors at offsets {-k, 0, k}
        for (int di = -k; di <= k; di += k) {
            for (int dj = -k; dj <= k; dj += k) {
                for (int dk = -k; dk <= k; dk += k) {
                    if (di == 0 && dj == 0 && dk == 0) continue; // Skip the center voxel

                    uvec3 q = p + uvec3(di, dj, dk);
                    if (any(greaterThanEqual(q, uvec3(chunk_size)))) continue; // Out of bounds

                    // Read neighbor seed position
                    vec4 qSeed = imageLoad(toggle ? positions1 : positions0, ivec3(q));

                    // If neighbor has a seed assigned
                    if (qSeed.w > 0.5) {
                        if (pSeed.w < 0.5) {
                            // Current voxel has no seed, adopt neighbor's seed
                            pSeed = qSeed;
                        } else {
                            // Both voxels have seeds, check distances
                            float distP = distance(vec3(p), pSeed.xyz);
                            float distQ = distance(vec3(p), qSeed.xyz);
                            if (distQ < distP) {
                                pSeed = qSeed;
                            }
                        }
                    }
                }
            }
        }

        // Write updated seed position to the other buffer
        imageStore(toggle ? positions0 : positions1, ivec3(p), pSeed);

        // Memory barrier before next iteration
        barrier();
        memoryBarrierImage();

        // Swap buffers
        toggle = !toggle;
    }

    // Distance Field Computation
    // --------------------------
    // After JFA iterations, compute the distance to the closest seed
    vec4 finalSeed = imageLoad(toggle ? positions1 : positions0, ivec3(p));
    float distanceToSeed = 0.0;
    if (finalSeed.w > 0.5) {
        distanceToSeed = distance(vec3(p), finalSeed.xyz);
    } else {
        // If no seed was found, assign a maximum distance
        distanceToSeed = float(chunk_size) * sqrt(3.0); // Maximum possible distance in the grid
    }

    // Write the distance to the distance field image
    imageStore(distanceField, ivec3(p), vec4(distanceToSeed));

    // Final memory barrier to ensure all writes are completed
    barrier();
    memoryBarrierImage();
}
