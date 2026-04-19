<script lang="ts">
	import type { Project } from '$lib/data/projects';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();
</script>

<a
	href={project.link}
	target="_blank"
	rel="noopener noreferrer"
	class="project-card"
	class:span-full={project.spanFull}
>
	<div class="project-img" style="background-image:url('{project.image}')"></div>
	<div class="project-text">
		<div class="project-type">{project.type}</div>
		<h3 class="project-name">{project.title}</h3>
		<p class="project-desc">{project.description}</p>
		<div class="project-stack">
			{#each project.tags as tag}
				<span>{tag}</span>
			{/each}
		</div>
	</div>
</a>

<style>
	.project-card {
		background: var(--surface);
		border: 1px solid var(--border);
		overflow: hidden;
		transition: all 0.4s var(--ease);
		position: relative;
		text-decoration: none;
		display: block;
	}

	.project-card:hover {
		transform: translateY(-3px);
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
	}

	.project-card::after {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 2px;
		background: var(--accent);
		opacity: 0;
		transition: opacity 0.4s;
	}

	.project-card:hover::after {
		opacity: 1;
	}

	.project-card.span-full {
		grid-column: 1 / -1;
	}

	.project-img {
		height: 140px;
		background-size: cover;
		background-position: center;
		position: relative;
	}

	.project-img::after {
		content: '';
		position: absolute;
		inset: 0;
		background: linear-gradient(180deg, transparent 30%, var(--bg) 100%);
		pointer-events: none;
	}

	.project-text {
		padding: 1.5rem 1.8rem;
	}

	.project-type {
		font-size: 0.5rem;
		font-weight: 500;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
		transition: color 0.5s;
	}

	.project-name {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: 1.05rem;
		color: var(--heading);
		margin-bottom: 0.4rem;
		line-height: 1.3;
		transition: color 0.5s;
	}

	.project-desc {
		font-size: 0.78rem;
		color: var(--text-dim);
		line-height: 1.6;
		margin-bottom: 0.8rem;
		transition: color 0.5s;
	}

	.project-stack {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
	}

	.project-stack span {
		font-size: 0.5rem;
		color: var(--text-muted);
		padding: 0.1rem 0.4rem;
		border: 1px solid var(--border);
		letter-spacing: 0.05em;
		transition: var(--theme-transition);
	}
</style>
