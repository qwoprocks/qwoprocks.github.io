<script lang="ts">
	import { onMount } from 'svelte';

	onMount(() => {
		const moduleScript = document.createElement('script');
		moduleScript.type = 'module';
		moduleScript.src = '/js/grass/main.js';
		document.head.appendChild(moduleScript);

		return () => {
			moduleScript.remove();
		};
	});
</script>

<svelte:head>
	<title>Gallery | Ming Chong Lim</title>
</svelte:head>
