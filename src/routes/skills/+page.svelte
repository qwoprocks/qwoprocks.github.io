<script lang="ts">
	interface SkillCategory {
		name: string;
		skills: { name: string; proficient: boolean }[];
	}

	const categories: SkillCategory[] = [
		{
			name: 'Programming Languages',
			skills: [
				{ name: 'Python', proficient: true },
				{ name: 'Java', proficient: true },
				{ name: 'C/C++', proficient: true },
				{ name: 'Javascript/Typescript', proficient: true },
				{ name: 'C#', proficient: false },
				{ name: 'Rust', proficient: false },
				{ name: 'PHP', proficient: false },
				{ name: 'PostgreSQL', proficient: false }
			]
		},
		{
			name: 'Machine Learning',
			skills: [
				{ name: 'PyTorch', proficient: true },
				{ name: 'TensorFlow', proficient: true },
				{ name: 'Keras', proficient: false },
				{ name: 'Scikit-Learn', proficient: false },
				{ name: 'OpenCV', proficient: false }
			]
		},
		{
			name: 'Frameworks',
			skills: [
				{ name: 'React.js', proficient: false },
				{ name: 'Vue.js', proficient: false },
				{ name: 'Ionic 5', proficient: false },
				{ name: 'JavaFX', proficient: false },
				{ name: 'Kivy', proficient: false },
				{ name: 'JQuery', proficient: false },
				{ name: 'HTML5', proficient: false },
				{ name: 'CSS3/SCSS', proficient: false }
			]
		},
		{
			name: 'Tools',
			skills: [
				{ name: 'Docker/Docker-Compose', proficient: true },
				{ name: 'Git', proficient: true },
				{ name: 'NumPy', proficient: false },
				{ name: 'Pandas', proficient: false },
				{ name: 'Regex', proficient: false }
			]
		}
	];
</script>

<svelte:head>
	<title>Skills | Ming Chong Lim</title>
</svelte:head>

<h1>Skills</h1>

<div class="skill-grid">
	{#each categories as category}
		<div class="skill-category">
			<h2>{category.name}</h2>
			<div class="skill-tags">
				{#each category.skills as skill}
					<span class="skill-tag" class:proficient={skill.proficient}>{skill.name}</span>
				{/each}
			</div>
		</div>
	{/each}
</div>

<div class="legend">
	<span class="legend-item"><span class="legend-chip proficient"></span> Proficient</span>
	<span class="legend-item"><span class="legend-chip"></span> Familiar</span>
</div>

<style>
	.skill-grid {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 2em;
		margin-bottom: 2em;
	}

	.skill-category h2 {
		font-size: 1em;
		font-weight: 600;
		margin-bottom: 0.8em;
		padding-bottom: 0.5em;
		border-bottom: 2px solid #4a5d3a;
		display: inline-block;
	}

	.skill-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}

	.skill-tag {
		font-size: 0.85em;
		padding: 6px 14px;
		border-radius: 6px;
		background: #f8f8f8;
		border: 1px solid #e5e7eb;
		color: #333;
		transition:
			transform 0.15s ease,
			box-shadow 0.15s ease;
	}

	.skill-tag:hover {
		transform: translateY(-1px);
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
	}

	.skill-tag.proficient {
		background: rgba(74, 93, 58, 0.1);
		border-color: rgba(74, 93, 58, 0.25);
		color: #4a5d3a;
		font-weight: 600;
	}

	.legend {
		display: flex;
		gap: 1.5em;
		font-size: 0.8em;
		color: #999;
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.legend-chip {
		display: inline-block;
		width: 12px;
		height: 12px;
		border-radius: 3px;
		background: #f8f8f8;
		border: 1px solid #e5e7eb;
	}

	.legend-chip.proficient {
		background: rgba(74, 93, 58, 0.1);
		border-color: rgba(74, 93, 58, 0.25);
	}

	@media screen and (max-width: 640px) {
		.skill-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
