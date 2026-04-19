<script lang="ts">
	import SectionHeader from '$lib/components/SectionHeader.svelte';
	import ShutterLine from '$lib/components/ShutterLine.svelte';
	import ProjectCard from '$lib/components/ProjectCard.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { projects } from '$lib/data/projects';
</script>

<svelte:head>
	<title>Projects | Ming Chong Lim</title>
</svelte:head>

<div class="container detail-page">
	<section class="section">
		<ShutterLine />
		<SectionHeader num="05" title="Projects" />

		<div class="projects-grid reveal reveal-d1" use:reveal>
			{#each projects as project}
				<ProjectCard {project} />
			{/each}
		</div>
	</section>

	<Footer />
</div>

<style>
	.detail-page {
		padding-top: 80px;
	}

	.projects-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.2rem;
	}

	@media (max-width: 768px) {
		.projects-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
