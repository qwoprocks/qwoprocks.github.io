<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';

	let theme = $state('dark');

	onMount(() => {
		const stored = localStorage.getItem('theme');
		if (stored) {
			theme = stored;
		}
		document.documentElement.setAttribute('data-theme', theme);
	});

	function toggle() {
		theme = theme === 'dark' ? 'light' : 'dark';
		document.documentElement.setAttribute('data-theme', theme);
		localStorage.setItem('theme', theme);
	}
</script>

<button class="theme-toggle" onclick={toggle} aria-label="Toggle theme">
	<svg class="moon" viewBox="0 0 24 24"
		><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg
	>
	<svg class="sun" viewBox="0 0 24 24"
		><circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line
			x1="12"
			y1="21"
			x2="12"
			y2="23"
		/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line
			x1="18.36"
			y1="18.36"
			x2="19.78"
			y2="19.78"
		/><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line
			x1="4.22"
			y1="19.78"
			x2="5.64"
			y2="18.36"
		/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></svg
	>
</button>

<style>
	.theme-toggle {
		width: 30px;
		height: 30px;
		border: 1px solid var(--border);
		border-radius: 50%;
		background: transparent;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: var(--theme-transition);
		flex-shrink: 0;
	}

	.theme-toggle:hover {
		border-color: var(--accent);
	}

	.theme-toggle svg {
		width: 13px;
		height: 13px;
		fill: none;
		stroke: var(--text-dim);
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
		transition: stroke 0.3s;
	}

	.theme-toggle:hover svg {
		stroke: var(--accent);
	}

	.theme-toggle .sun {
		display: none;
	}

	.theme-toggle .moon {
		display: block;
	}

	:global([data-theme='light']) .theme-toggle .sun {
		display: block;
	}

	:global([data-theme='light']) .theme-toggle .moon {
		display: none;
	}
</style>
