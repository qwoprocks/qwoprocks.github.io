<script lang="ts">
	import SectionHeader from '$lib/components/SectionHeader.svelte';
	import ShutterLine from '$lib/components/ShutterLine.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { awards } from '$lib/data/awards';
</script>

<svelte:head>
	<title>Awards | Ming Chong Lim</title>
</svelte:head>

<div class="container detail-page">
	<section class="section">
		<ShutterLine />
		<SectionHeader num="04" title="Awards" />

		{#each awards as award}
			<div class="award-row reveal" use:reveal>
				<div class="award-place">{award.place}</div>
				<div class="award-info">
					<h3>{award.title}</h3>
					<div class="award-date">{award.date}</div>
					<p>{award.description}</p>
					{#if award.details}
						<ul class="award-details">
							{#each award.details as detail}
								<li>{@html detail}</li>
							{/each}
						</ul>
					{/if}
				</div>
			</div>
		{/each}
	</section>

	<Footer />
</div>

<style>
	.detail-page {
		padding-top: 80px;
	}

	.award-row {
		display: grid;
		grid-template-columns: 40px 1fr;
		gap: 1.2rem;
		padding: 1.5rem 0;
		border-bottom: 1px solid var(--border);
		transition: all 0.3s;
	}

	.award-row:last-of-type {
		border-bottom: none;
	}

	.award-row:hover {
		padding-left: 0.3rem;
	}

	.award-place {
		font-family: var(--font-head);
		font-weight: 300;
		font-size: 1.3rem;
		color: var(--accent);
		font-style: italic;
		transition: color 0.5s;
	}

	.award-info h3 {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: 0.88rem;
		color: var(--heading);
		margin-bottom: 0.1rem;
		transition: color 0.5s;
	}

	.award-date {
		font-size: 0.6rem;
		color: var(--text-muted);
		letter-spacing: 0.05em;
		margin-bottom: 0.3rem;
		transition: color 0.5s;
	}

	.award-info p {
		font-size: 0.72rem;
		color: var(--text-dim);
		transition: color 0.5s;
	}

	.award-details {
		list-style: none;
		margin-top: 0.8rem;
		padding: 0;
	}

	.award-details li {
		font-size: 0.72rem;
		color: var(--text-dim);
		line-height: 1.7;
		padding: 0.15rem 0;
		padding-left: 1rem;
		position: relative;
		transition: color 0.5s;
	}

	.award-details li::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0.5rem;
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--border-light);
	}

	.award-details :global(a) {
		color: var(--accent);
		text-decoration: none;
		border-bottom: 1px solid var(--accent-soft);
		transition: border-color 0.3s, color 0.5s;
	}

	.award-details :global(a:hover) {
		border-color: var(--accent);
	}
</style>
