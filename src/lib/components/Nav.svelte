<script lang="ts">
	import { page } from '$app/state';

	interface NavItem {
		title: string;
		url: string;
		subnav?: NavItem[];
	}

	interface Props {
		nav: NavItem[];
	}

	let { nav }: Props = $props();
</script>

<ul class="nav">
	{#each nav as item}
		<li class:active={page.url.pathname === item.url}>
			<a href={item.url}>{item.title}</a>
		</li>

		{#if item.subnav}
			<Nav nav={item.subnav} />
		{/if}
	{/each}
</ul>

<style>
	.nav {
		display: flex;
		flex-direction: column;
		gap: 2.2vh;
		padding: 0;
		margin-top: 22px;
		margin-bottom: 35px;
		font-size: 1.1em;
	}

	.nav li {
		list-style-type: none;
	}

	.nav li.active a {
		font-weight: bold;
		color: #555d47;
		cursor: default;
	}

	.nav li.active a::after {
		display: none;
		transform: translate3d(0, 0, 0);
	}

	.nav li a {
		display: inline-block;
		color: #6c765b;
		position: relative;
		padding: 0.2em 0;
		overflow: hidden;
	}

	.nav li a::after {
		content: '';
		position: absolute;
		bottom: 0;
		left: 0;
		width: 100%;
		height: 0.1em;
		background-color: #6c765b;
		transition: opacity 200ms, transform 200ms;
		opacity: 1;
		transform: translate3d(-100%, 0, 0);
	}

	.nav li a:hover::after,
	.nav li a:focus::after {
		transform: translate3d(0, 0, 0);
	}
</style>
