<script lang="ts">
	import SectionHeader from '$lib/components/SectionHeader.svelte';
	import ShutterLine from '$lib/components/ShutterLine.svelte';
	import TimelineItem from '$lib/components/TimelineItem.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { experiences } from '$lib/data/experiences';
</script>

<svelte:head>
	<title>Experience | Ming Chong Lim</title>
</svelte:head>

<div class="container detail-page">
	<section class="section">
		<ShutterLine />
		<SectionHeader num="02" title="Experience" />

		<div class="timeline reveal reveal-d1" use:reveal>
			{#each experiences as exp}
				<div class="tl-item-full">
					<TimelineItem experience={exp} showDesc={false} />
					<ul class="tl-bullets">
						{#each exp.bullets as bullet}
							<li>{bullet}</li>
						{/each}
					</ul>
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

	.timeline {
		position: relative;
		padding-left: 2rem;
	}

	.timeline::before {
		content: '';
		position: absolute;
		left: 4px;
		top: 8px;
		bottom: 8px;
		width: 1px;
		background: linear-gradient(to bottom, var(--border-light), var(--border), transparent);
		transition: background 0.5s;
	}

	.tl-bullets {
		padding: 0 2rem;
		margin-bottom: 1.5rem;
		list-style: none;
	}

	.tl-bullets li {
		font-size: var(--text-lg);
		color: var(--text-dim);
		line-height: 1.7;
		padding: 0.15rem 0;
		position: relative;
		padding-left: 1rem;
		transition: color 0.5s;
	}

	.tl-bullets li::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0.55rem;
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--border-light);
	}

	@media (max-width: 480px) {
		.timeline {
			padding-left: 1.5rem;
		}

		.tl-bullets {
			padding-left: 1.5rem;
		}
	}
</style>
