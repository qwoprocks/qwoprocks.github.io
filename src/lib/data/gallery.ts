export interface GalleryWriteupSection {
	title: string;
	content: string;
}

export interface GalleryItem {
	id: string;
	title: string;
	description: string;
	date: string;
	tags: string[];
	scriptSrc?: string;
	sections: GalleryWriteupSection[];
	references: { label: string; url: string }[];
}

export const galleryItems: GalleryItem[] = [
	{
		id: 'grass-scene',
		title: 'Procedural Grassland',
		description:
			'A real-time procedural grassland with volumetric clouds, mountains, and 100k instanced grass blades — all procedural, no pre-baked assets. Try the post-processing filters above the writeup to see Kuwahara, spread, pixelate, and edge detection effects applied in real time.',
		date: '2024',
		tags: ['Three.js', 'GLSL', 'Procedural Generation', 'WebGL'],
		sections: [
			{
				title: 'Overview',
				content: `<p>The scene has five elements: a sky dome, terrain plane, instanced grass, volumetric clouds, and a mountain range. Everything is procedural. The pipeline targets 30 fps with frame-rate compensation.</p>`
			},
			{
				title: 'Grass Rendering',
				content: `<p>100,000 blades are rendered using a single <code>InstancedMesh</code>. Each blade is a 2D <code>ShapeGeometry</code> defined by bezier curves, keeping vertex count low while maintaining a natural tapered shape.</p>
<p>Positions are sampled from the terrain using a custom <code>DistanceBasedMeshSurfaceSampler</code> that extends Three.js's <code>MeshSurfaceSampler</code>. It biases density toward the camera via inverse-distance weighting: <code>1 / (d^0.96 + 1)</code>, so the foreground gets the densest coverage. Blades outside the frustum are rejected at placement time.</p>
<p>Wind is done entirely in the vertex shader. Each blade is bent along a circular arc parameterized by inverse radius <code>r_inv</code>, driven by three octaves of simplex noise FBM. A time-varying noise layer makes the wind shift over time. The math maps each vertex height <code>k</code> to <code>(sin(k * r_inv), cos(k * r_inv))</code>.</p>
<p>Per-instance color comes from a hand-picked palette interpolated via Perlin noise FBM at each blade's world position. Lighting uses Blinn-Phong with normals recomputed per-fragment via <code>dFdx</code>/<code>dFdy</code> derivatives, since the vertex shader displacement invalidates the originals. Cloud shadows are sampled from the 3D noise texture.</p>`
			},
			{
				title: 'Cloud System',
				content: `<p>Clouds use GPU-computed volumetric rendering. A custom <code>GPUComputationRenderer3D</code> generates a 256&times;256&times;256 3D noise texture slice-by-slice. Each slice combines Perlin and Worley noise:</p>
<ol>
<li>Billowy Perlin: 7-octave FBM, remapped with <code>abs(x * 2 - 1)</code> for billowy shapes</li>
<li>Worley FBM: 4 layers of tileable Worley noise at increasing frequencies</li>
<li>Final density: <code>remap(perlinWorley, worleyFBM - 1, 1, 0, 1)</code> with coverage cutoff</li>
</ol>
<p>This follows Andrew Schneider's <em>Real-Time Volumetric Cloudscapes</em> from GPU Pro 7.</p>
<p>Rendering ray-marches through a box volume in 15 steps, sampling the 3D texture at each point. The noise scrolls over time for drift. A height gradient sculpts density &mdash; thinner near top and bottom. Jitter via Wang hash reduces banding. Shading uses directional derivatives (sampling density at two offset points) to approximate light scattering.</p>`
			},
			{
				title: 'Mountains',
				content: `<p>The mountain range uses <em>Diffusion-Limited Aggregation</em> (DLA) to generate a heightmap. Particles undergo random walks until they contact an existing cluster, forming fractal branching structures. The result is iteratively upscaled (7 expansion passes, doubling resolution each time with random perturbation). Each pass accumulates onto a canvas, blending layers.</p>
<p>A radial falloff tapers the edges. The texture serves as both color map and displacement map on a subdivided plane (200&times;200 segments), creating a mountain silhouette at the horizon.</p>`
			},
			{
				title: 'Sky and Terrain',
				content: `<p>The sky is an inverted sphere with a vertical gradient (dark blue top, light blue horizon), rendered with <code>MeshBasicMaterial</code> so fog doesn't affect it.</p>
<p>Terrain is a 10,000&times;10,000 unit subdivided plane with vertex displacement from low-frequency Perlin noise, creating gentle rolling hills. Blinn-Phong shading with a warm sandy color. Distance fog (<code>THREE.Fog</code>) between 5,450&ndash;5,550 units softens the terrain-sky transition.</p>`
			},
			{
				title: 'Post-Processing',
				content: `<p>Rendering goes through an <code>EffectComposer</code> pipeline with four toggleable <code>ShaderPass</code> filters:</p>
<ol>
<li><strong>Kuwahara</strong> &mdash; Edge-preserving smoothing that produces a painterly/oil-painting look. Computes mean and variance across four quadrants of a local window, picking the quadrant with lowest variance.</li>
<li><strong>Spread</strong> &mdash; Replaces each pixel with a random nearby pixel within a configurable radius, creating a scattered/diffused texture reminiscent of pointillist brushwork.</li>
<li><strong>Pixelate</strong> &mdash; Snaps UV coordinates to a grid, producing a mosaic/pixel-art effect.</li>
<li><strong>Edge Detection</strong> &mdash; Sobel operator computing horizontal and vertical gradients, blended with the original image to overlay detected edges.</li>
</ol>
<p>A <code>ResizeObserver</code> dynamically adjusts the renderer to match the container width while maintaining a 4:3 aspect ratio. Filters can be toggled via the controls above the scene.</p>`
			}
		],
		references: [
			{
				label: 'GPU Pro 7: Real-Time Volumetric Cloudscapes (A. Schneider)',
				url: 'https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn'
			},
			{
				label: 'Simplex Noise (Ashima Arts / stegu)',
				url: 'https://github.com/ashima/webgl-noise'
			},
			{
				label: 'Hash without Sine (David Hoskins, Shadertoy)',
				url: 'https://www.shadertoy.com/view/3dVXDc'
			},
			{
				label: 'Diffusion-Limited Aggregation',
				url: 'https://en.wikipedia.org/wiki/Diffusion-limited_aggregation'
			},
			{
				label: 'Three.js InstancedMesh',
				url: 'https://threejs.org/docs/#api/en/objects/InstancedMesh'
			},
			{
				label: 'Three.js MeshSurfaceSampler',
				url: 'https://threejs.org/docs/#examples/en/math/MeshSurfaceSampler'
			}
		]
	},
	{
		id: 'maze-generator',
		title: 'Conway Maze Generator',
		description:
			'A maze generator based on a modified Conway\'s Game of Life ruleset. Draw an initial seed pattern or use the default, then watch the cellular automata evolve into a maze structure. Did this as part of the bonus section of a maze solving project during a undergraduate CS course.',
		date: '2020',
		tags: ['Cellular Automata', 'Canvas', 'Procedural Generation'],
		scriptSrc: '/js/maze/maze.js',
		sections: [
			{
				title: 'How It Works',
				content:
					'<p>The generator uses a modified Game of Life ruleset drawing inspiration from <a href="https://www.reddit.com/r/proceduralgeneration/comments/7v9jbj/" target="_blank" rel="noopener noreferrer">this Reddit post</a>. Starting from a small cross-shaped seed at the center, the automata evolves through several steps:</p><ol><li>Cells with 0 neighbours die (isolation)</li><li>Cells with 1-5 neighbours survive, with a special arch-breaking rule that prevents floating structures</li><li>Cells with exactly 4 neighbours trigger diagonal kills to create passages</li><li>Cells with more than 5 neighbours die (overcrowding)</li><li>Dead cells with exactly 2 neighbours are born</li></ol><p>After the simulation runs for <code>rows + cols - 30</code> steps, outer walls are built around the perimeter.</p>'
			},
			{
				title: 'Implementation',
				content:
					'<p>The original was written in Java. This port runs entirely on a canvas element with requestAnimationFrame. The grid state is stored as a 2D boolean array. Each simulation step creates a fresh next-state array to avoid mutation artifacts during rule evaluation.</p><p>The draw mode lets you paint initial conditions before running the automata. The canvas handles both click and drag events, mapping mouse coordinates to grid cells. Three fixed grid sizes (20&times;20, 50&times;50, 100&times;100) are available — smaller grids produce thicker, simpler mazes while larger grids create more intricate structures.</p>'
			}
		],
		references: [
			{
				label: "Reddit: Procedurally Generated Maze Based on Conway's Game of Life",
				url: 'https://www.reddit.com/r/proceduralgeneration/comments/7v9jbj/'
			},
			{
				label: "Conway's Game of Life (Wikipedia)",
				url: 'https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life'
			}
		]
	}
];
