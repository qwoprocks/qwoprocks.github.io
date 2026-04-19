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
	<link rel="shortcut icon" type="image/x-icon" href="/favicon.svg" />
	<title>Ming Chong Lim — Quantitative Developer</title>
	<meta name="description" content="Ming Chong Lim — Quantitative Developer at DRW. Carnegie Mellon ML graduate. Portfolio showcasing experience, projects, and research in machine learning, systems programming, and quantitative development." />
	<meta property="og:title" content="Ming Chong Lim — Quantitative Developer" />
	<meta property="og:description" content="Quantitative Developer at DRW. Carnegie Mellon ML graduate. Portfolio showcasing experience, projects, and research." />
	<meta property="og:type" content="website" />
	<meta property="og:url" content="https://qwoprocks.github.io/" />
	<meta name="twitter:card" content="summary" />
	<meta name="twitter:title" content="Ming Chong Lim — Quantitative Developer" />
	<meta name="twitter:description" content="Quantitative Developer at DRW. Carnegie Mellon ML graduate." />
</svelte:head>

<Nav />
{@render children()}

<style>
	:global(a) {
		color: inherit;
	}
</style>
