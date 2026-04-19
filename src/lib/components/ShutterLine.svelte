<script lang="ts">
	import { shutter } from '$lib/actions/reveal';
</script>

<div class="shutter" use:shutter></div>

<style>
	.shutter {
		width: 0;
		height: 1px;
		background: var(--border-light);
		margin: 0 auto 3rem;
		transition: width 1.2s var(--ease), background 0.5s;
	}

	:global(.shutter.open) {
		width: 100%;
	}
</style>
