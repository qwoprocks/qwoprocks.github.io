<script lang="ts">
	import '$lib/styles/theme.css';
	import Nav from '$lib/components/Nav.svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();
</script>

<svelte:head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<link rel="shortcut icon" type="image/x-icon" href="/favicon.svg" />
	<title>Ming Chong Lim</title>
</svelte:head>

<Nav />
{@render children()}

<style>
	:global(a) {
		color: inherit;
	}
</style>
