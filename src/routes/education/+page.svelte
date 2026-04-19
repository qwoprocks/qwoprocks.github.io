<script lang="ts">
	import SectionHeader from '$lib/components/SectionHeader.svelte';
	import ShutterLine from '$lib/components/ShutterLine.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { education } from '$lib/data/education';
</script>

<svelte:head>
	<title>Education | Ming Chong Lim</title>
</svelte:head>

<div class="container detail-page">
	<section class="section">
		<ShutterLine />
		<SectionHeader num="03" title="Education" />

		{#each education as edu, i}
			<div class="edu-row reveal" use:reveal>
				<div class="edu-year">{edu.period}</div>
				<div class="edu-divider"></div>
				<div class="edu-info">
					<h3>{edu.school}</h3>
					<div class="degree">{edu.degree}</div>
					{#if edu.gpa}
						<div class="gpa">GPA: {edu.gpa}</div>
					{/if}
					<div class="note">{edu.note}</div>
					<ul class="edu-details">
						{#each edu.details as detail}
							<li>{@html detail}</li>
						{/each}
					</ul>
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

	.edu-row {
		display: grid;
		grid-template-columns: 100px 1px 1fr;
		gap: 1.5rem;
		padding: 1.5rem 0;
		align-items: start;
		transition: var(--theme-transition);
	}

	.edu-row + .edu-row {
		border-top: 1px solid var(--border);
	}

	.edu-year {
		font-size: 0.68rem;
		color: var(--text-muted);
		text-align: right;
		padding-top: 0.15rem;
	}

	.edu-divider {
		width: 1px;
		height: 100%;
		background: var(--border-light);
		min-height: 40px;
		transition: background 0.5s;
	}

	.edu-info h3 {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: 1rem;
		color: var(--heading);
		margin-bottom: 0.1rem;
		transition: color 0.5s;
	}

	.degree {
		font-size: 0.8rem;
		color: var(--accent);
		font-weight: 400;
		margin-bottom: 0.2rem;
		font-family: var(--font-head);
		transition: color 0.5s;
	}

	.gpa {
		font-size: 0.72rem;
		color: var(--heading);
		font-weight: 500;
		margin-bottom: 0.2rem;
		transition: color 0.5s;
	}

	.note {
		font-size: 0.72rem;
		color: var(--text-dim);
		transition: color 0.5s;
	}

	.edu-details {
		list-style: none;
		margin-top: 0.8rem;
		padding: 0;
	}

	.edu-details li {
		font-size: 0.72rem;
		color: var(--text-dim);
		line-height: 1.7;
		padding: 0.1rem 0;
		padding-left: 1rem;
		position: relative;
		transition: color 0.5s;
	}

	.edu-details li::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0.5rem;
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--border-light);
	}

	.edu-details :global(a) {
		color: var(--accent);
		text-decoration: none;
		border-bottom: 1px solid var(--accent-soft);
		transition: border-color 0.3s, color 0.5s;
	}

	.edu-details :global(a:hover) {
		border-color: var(--accent);
	}

	@media (max-width: 768px) {
		.edu-row {
			grid-template-columns: 1fr;
			gap: 0.5rem;
		}

		.edu-year {
			text-align: left;
		}

		.edu-divider {
			display: none;
		}
	}
</style>
