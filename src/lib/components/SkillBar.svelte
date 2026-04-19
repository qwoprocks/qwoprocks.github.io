<script lang="ts">
	import type { Skill } from '$lib/data/skills';
	import { reveal } from '$lib/actions/reveal';

	interface Props {
		skill: Skill;
	}

	let { skill }: Props = $props();
</script>

<div class="spec-item" style="--w:{skill.level}%" use:reveal={{ threshold: 0.3 }}>
	<span class="name" class:pro={skill.proficient}>{skill.name}</span>
	<div class="line"><div class="progress"></div></div>
</div>

<style>
	.spec-item {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-bottom: 0.55rem;
	}

	.name {
		font-size: var(--text-lg);
		color: var(--text-dim);
		width: 85px;
		flex-shrink: 0;
		transition: color 0.5s;
	}

	.name.pro {
		color: var(--text);
	}

	.line {
		flex: 1;
		height: 1px;
		background: var(--border);
		position: relative;
		transition: background 0.5s;
	}

	.progress {
		height: 1px;
		background: var(--heading);
		width: 0;
		transition: width 1s var(--ease), background 0.5s;
		position: absolute;
		left: 0;
		top: 0;
	}

	.progress::after {
		content: '';
		position: absolute;
		right: -2px;
		top: -2px;
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--heading);
		opacity: 0;
		transition: opacity 0.3s 1s, background 0.5s;
	}

	:global(.spec-item.visible) .progress {
		width: var(--w);
	}

	:global(.spec-item.visible) .progress::after {
		opacity: 1;
	}
</style>
