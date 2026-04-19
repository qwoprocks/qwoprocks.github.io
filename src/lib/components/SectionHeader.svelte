<script lang="ts">
	import { reveal } from '$lib/actions/reveal';

	interface Props {
		num: string;
		title: string;
	}

	let { num, title }: Props = $props();
</script>

<div class="waypoint reveal" use:reveal>
	<div class="waypoint-diamond"><span>{num}</span></div>
	<h2 class="waypoint-title">{title}</h2>
</div>

<style>
	.waypoint {
		display: inline-flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 2.5rem;
	}

	.waypoint-diamond {
		width: 26px;
		height: 26px;
		border: 1.5px solid var(--border-light);
		transform: rotate(45deg);
		display: flex;
		align-items: center;
		justify-content: center;
		transition: all 0.5s var(--ease);
	}

	.waypoint-diamond span {
		transform: rotate(-45deg);
		font-family: var(--font-mono);
		font-size: 0.55rem;
		font-weight: 400;
		color: var(--text-muted);
		letter-spacing: 0.05em;
		transition: color 0.5s;
	}

	.waypoint :global(.waypoint-diamond) {
		transition: all 0.5s var(--ease);
	}

	:global(.waypoint.visible) .waypoint-diamond {
		border-color: var(--accent);
		box-shadow: 0 0 12px var(--accent-dim);
	}

	:global(.waypoint.visible) .waypoint-diamond span {
		color: var(--accent);
	}

	.waypoint-title {
		font-family: var(--font-head);
		font-weight: 400;
		font-size: 1.6rem;
		color: var(--heading);
		letter-spacing: -0.02em;
		transition: color 0.5s;
	}
</style>
