<script lang="ts">
	import type { Experience } from '$lib/data/experiences';

	interface Props {
		experience: Experience;
		showDesc?: boolean;
		expandable?: boolean;
	}

	let { experience, showDesc = true, expandable = false }: Props = $props();
	let expanded = $state(false);

	function toggle() {
		if (expandable) expanded = !expanded;
	}
</script>

<div
	class="tl-item"
	class:expandable
	class:expanded
	id={experience.id}
	onclick={toggle}
	onkeydown={(e) => e.key === 'Enter' && toggle()}
	role={expandable ? 'button' : undefined}
	tabindex={expandable ? 0 : undefined}
>
	<div class="tl-content">
		<div class="tl-period">{experience.period}</div>
		<h3>{experience.title}</h3>
		<div class="tl-company">{experience.company}</div>
		{#if showDesc && experience.desc}
			<p class="tl-desc">{experience.desc}</p>
		{/if}
		{#if expandable}
			<div class="tl-detail-hint">
				<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
					<polyline points="6 9 12 15 18 9" />
				</svg>
			</div>
			<div class="tl-expand">
				<div class="tl-expand-inner">
					<ul class="tl-bullets">
						{#each experience.bullets as bullet}
							<li>{bullet}</li>
						{/each}
					</ul>
				</div>
			</div>
		{/if}
	</div>
	<span class="exp-tag">{experience.tag}</span>
</div>

<style>
	.tl-item {
		position: relative;
		padding: 1.5rem;
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}

	.tl-item::before {
		content: '';
		position: absolute;
		left: -2rem;
		top: 1.85rem;
		width: 9px;
		height: 9px;
		border-radius: 50%;
		background: var(--bg);
		border: 1.5px solid var(--accent);
		z-index: 1;
		transition: all 0.3s;
	}

	.tl-item:hover::before {
		background: var(--accent);
		box-shadow: 0 0 10px var(--accent-dim);
	}

	.tl-content {
		flex: 1;
		min-width: 0;
	}

	.tl-period {
		font-size: var(--text-base);
		color: var(--text-muted);
		margin-bottom: 0.3rem;
		letter-spacing: 0.02em;
		transition: color 0.5s;
	}

	h3 {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: var(--text-2xl);
		color: var(--heading);
		margin-bottom: 0.1rem;
		transition: color 0.5s;
	}

	.tl-company {
		font-size: var(--text-lg);
		color: var(--text-dim);
		font-family: var(--font-head);
		transition: color 0.5s;
	}

	.tl-desc {
		font-size: var(--text-lg);
		color: var(--text-muted);
		margin-top: 0.3rem;
		transition: color 0.5s;
	}

	/* Expandable hint */
	.tl-detail-hint {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		margin-top: 0.5rem;
	}

	.chevron {
		width: 14px;
		height: 14px;
		color: var(--text-muted);
		transition: transform 0.35s var(--ease), color 0.3s;
	}

	.expanded .chevron {
		transform: rotate(180deg);
	}

	.expandable {
		cursor: pointer;
		transition: background 0.3s;
	}

	.expandable:hover {
		background: var(--accent-dim);
	}

	.expandable:hover .chevron {
		color: var(--accent);
	}

	/* Expand animation using grid trick */
	.tl-expand {
		display: grid;
		grid-template-rows: 0fr;
		transition: grid-template-rows 0.4s var(--ease);
	}

	.expanded .tl-expand {
		grid-template-rows: 1fr;
	}

	.tl-expand-inner {
		overflow: hidden;
		min-height: 0;
	}

	.tl-bullets {
		list-style: none;
		padding: 0.6rem 0 0;
		margin: 0;
	}

	.tl-bullets li {
		font-size: var(--text-md);
		color: var(--text-dim);
		line-height: 1.7;
		padding: 0.15rem 0;
		padding-left: 1rem;
		position: relative;
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

	.exp-tag {
		font-size: var(--text-2xs);
		font-weight: 500;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--text-muted);
		padding: 0.2rem 0.6rem;
		border: 1px solid var(--border);
		height: fit-content;
		flex-shrink: 0;
		white-space: nowrap;
		transition: var(--theme-transition);
	}

	@media (max-width: 768px) {
		.tl-item {
			flex-direction: column;
		}

		.exp-tag {
			align-self: flex-start;
		}
	}
</style>
