<script lang="ts">
	import SectionHeader from '$lib/components/SectionHeader.svelte';
	import ShutterLine from '$lib/components/ShutterLine.svelte';
	import SkillBar from '$lib/components/SkillBar.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { skillGroups } from '$lib/data/skills';
</script>

<svelte:head>
	<title>Skills | Ming Chong Lim</title>
</svelte:head>

<div class="container detail-page">
	<section class="section">
		<ShutterLine />
		<SectionHeader num="06" title="Skills" />

		<div class="skills-spec reveal reveal-d1" use:reveal>
			{#each skillGroups as group}
				<div class="spec-group">
					<h3>{group.title}</h3>
					{#each group.skills as skill}
						<SkillBar {skill} />
					{/each}
				</div>
			{/each}
		</div>
	</section>

	<Footer />
</div>

<style>
	.detail-page {
		padding-top: 80px;
	}

	.skills-spec {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 3rem;
	}

	.spec-group h3 {
		font-size: var(--text-base);
		font-weight: 500;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--text-muted);
		margin-bottom: 1.2rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--border);
		transition: var(--theme-transition);
	}

	@media (max-width: 768px) {
		.skills-spec {
			grid-template-columns: 1fr;
		}
	}
</style>
