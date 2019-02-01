# Thread Block Sync

This repo contains an alternative approach to the spatial partitioning FRNNs search algorithm. The algorithm accesses messages, such that each CUDA block corresponds to a bin which the environment has been subdivided into. This technique is visible in molecular dynamics papers.

This requires that agents are ordered, however allows all threads within each block to act in sync sharing the workload of loading messages from texture memory (into shared).

The code runs the basic average model or FlameGPUs Circles model, which is intended as a simple particle model analogue. There is no visualisation, however it validates that results are in agreement with a control implementation.

## Testing notes

Notably, this technique requires threads be in order of message storage, as each threadblock must access the same bin (we could use a mapping, but that adds further expense).

Produced two implementation:
  * Basic, whereby a bin of messages is loaded at a time, followed by synchronisation.
  * Segmented, whereby all threads work to load all messages that will fit into shared memory.
  
I found that surprisingly, segmented performs better when shared memory is capped to 96 message per block across a range of densities, although this makes less of a difference at densities above 200 per radial neighbourhood.

Segmented performs roughly equal in 2D uniform random init whereby average radial neighbourhood is 100 agents (58 agents per bin). This may make sense as previously we have found 64 to be the minimum effective block size (although minimum block size is actually 96 due to our shared mem cap).
Segmented performance improves significantly as density increases above this level.

Profiling shows that occupancy is at 92%, bounded by registers per block. Register usage is fairly low (3072/65536). Surprisingly shared memory usage is only 792 bytes per block, allowing a block limit of 96, higher than the fixed limit of 32.

PC sampling shows that 47% of stalls are due to syncs, this is to be expected. The following highest are execution dependency (21%), other (11%) and instruction fetch (9%). Execution dependency appears tied to when messages are pulled from shared memory, hence unavoidable, unless we preload messages in a flipflop fashion?

In 3D, we find performance is roughly equal with 80 radial neighbours (under uniform random distribution), increasing as density increases. Appears the 96 messages rule holds true here too. Possible room for a minor optimisation, whereby all threads sum the 9 stripCount values independently, however this would require an additional sync.

### **TODO:** how does message distribution affect perf?
Considered doing this with some kind of perlin noise controlled random spawn. Sample perlin noise in a regular grid, for each grid cell randomly distribute N agents according to perlin noise value. Potential bias to results if regular grid aligned with FRNNs grid?
