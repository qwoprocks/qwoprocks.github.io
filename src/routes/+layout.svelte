<script lang="ts">
	import Nav from '$lib/components/Nav.svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	const nav = [
		{ title: 'Education', url: '/education' },
		{ title: 'Work Experience', url: '/experience' },
		{ title: 'Awards and Competitions', url: '/awards' },
		{ title: 'Projects', url: '/projects' },
		{ title: 'Skills', url: '/skills' },
		{ title: 'Gallery', url: '/gallery' }
	];
</script>

<svelte:head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<link rel="shortcut icon" type="image/x-icon" href="/favicon.svg" />
	<title>Ming Chong Lim</title>
</svelte:head>

<div class="wrapper">
	<header>
		<hr />
		<h2><a href="/">Ming Chong Lim</a></h2>
		<hr />

		<Nav {nav} />

		<div class="social-links">
			<a href="https://github.com/qwoprocks" target="_blank" title="GitHub" class="social-icon">
				<!-- Simple Icons: GitHub -->
				<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
					<path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
				</svg>
			</a>
			<a href="https://www.linkedin.com/in/ming-chong-lim/" target="_blank" title="LinkedIn" class="social-icon">
				<!-- Simple Icons: LinkedIn -->
				<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
					<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
				</svg>
			</a>
			<a href="mailto:limmingchong98@gmail.com" target="_blank" title="Email" class="social-icon">
				<!-- Envelope icon -->
				<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
					<path d="M1.5 8.67v8.58a3 3 0 0 0 3 3h15a3 3 0 0 0 3-3V8.67l-8.928 5.493a3 3 0 0 1-3.144 0L1.5 8.67Z" />
					<path d="M22.5 6.908V6.75a3 3 0 0 0-3-3h-15a3 3 0 0 0-3 3v.158l9.714 5.978a1.5 1.5 0 0 0 1.572 0L22.5 6.908Z" />
				</svg>
			</a>
			<a href="/Ming Chong Lim_Resume.pdf" target="_blank" title="Resume" class="social-icon">
				<!-- Document icon -->
				<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
					<path d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V8.625L14.25 1.5H5.625ZM14.25 1.5v5.25c0 1.035.84 1.875 1.875 1.875h5.25" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
					<path d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V8.625L14.25 1.5H5.625Z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
					<path d="M8.25 13.5h7.5M8.25 16.5h7.5M8.25 10.5h3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
				</svg>
			</a>
		</div>
	</header>
	<section>
		{@render children()}
	</section>
	<footer>
		<p><small>This site is always a work-in-progress, just like my life</small></p>
		<small>&copy; 2024 Ming Chong Lim</small>
	</footer>
</div>

<style>
	:global(body) {
		background-color: #fff;
		padding: 50px;
		font: 15px/1.5 'Noto Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
		color: #727272;
		font-weight: 400;
		margin: 0;
	}

	:global(h1, h2, h3, h4, h5, h6) {
		color: #222;
		margin: 0 0 20px;
	}

	:global(p, ul, ol, table, pre, dl) {
		margin: 0 0 20px;
	}

	:global(h1, h2, h3) {
		line-height: 1.1;
	}

	:global(h1:nth-child(n + 2)) {
		font-size: 28px;
		margin-top: 1.9em;
	}

	:global(h2) {
		color: #393939;
	}

	:global(h3, h4, h5, h6) {
		color: #494949;
	}

	:global(a) {
		color: #267cb9;
		text-decoration: none;
	}

	:global(a:hover, a:focus) {
		color: #047;
		font-weight: inherit;
	}

	:global(a small) {
		font-size: 11px;
		color: #777;
		margin-top: -0.3em;
		display: block;
	}

	:global(a:hover small) {
		color: #777;
	}

	:global(strong) {
		color: #222;
		font-weight: 700;
	}

	:global(img) {
		max-width: 100%;
	}

	:global(small) {
		font-size: 11px;
	}

	:global(hr) {
		border: 0;
		background: #e5e5e5;
		height: 1px;
		margin: 0 0 20px;
	}

	:global(blockquote) {
		border-left: 1px solid #e5e5e5;
		margin: 0;
		padding: 0 0 0 20px;
		font-style: italic;
	}

	:global(code, pre) {
		font-family: Monaco, Bitstream Vera Sans Mono, Lucida Console, Terminal, Consolas, Liberation Mono,
			DejaVu Sans Mono, Courier New, monospace;
		color: #333;
	}

	:global(pre) {
		padding: 8px 15px;
		background: #f8f8f8;
		border-radius: 5px;
		border: 1px solid #e5e5e5;
		overflow-x: auto;
	}

	:global(table) {
		width: 100%;
		border-collapse: collapse;
	}

	:global(th, td) {
		text-align: left;
		padding: 5px 10px;
		border-bottom: 1px solid #e5e5e5;
	}

	:global(dt) {
		color: #444;
		font-weight: 700;
	}

	:global(th) {
		color: #444;
	}

	:global(kbd) {
		background-color: #fafbfc;
		border: 1px solid #c6cbd1;
		border-bottom-color: #959da5;
		border-radius: 3px;
		box-shadow: inset 0 -1px 0 #959da5;
		color: #444d56;
		display: inline-block;
		font-size: 11px;
		line-height: 10px;
		padding: 3px 5px;
		vertical-align: middle;
	}

	/* External link icon */
	:global(section a[href^='http']:not(.noicon)::after),
	:global(section a[href^='https://']:not(.noicon)::after) {
		content: '';
		width: 11px;
		height: 11px;
		margin-left: 4px;
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' viewBox='0 0 16 16'%3E%3Cpath fill-rule='evenodd' d='M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5z'/%3E%3Cpath fill-rule='evenodd' d='M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0v-5z'/%3E%3C/svg%3E");
		background-position: center;
		background-repeat: no-repeat;
		background-size: contain;
		display: inline-block;
	}

	.wrapper {
		width: min(1024px, 100%);
		margin: 0 auto;
	}

	header {
		width: 270px;
		float: left;
		position: fixed;
		display: flex;
		flex-direction: column;
		-webkit-font-smoothing: subpixel-antialiased;
		min-height: calc(100vh - 80px);
	}

	header hr {
		height: 0.2em;
		width: 28%;
		background-color: #646e53;
	}

	header h2 a {
		color: #6c765b;
	}

	header h2 a:hover {
		color: #565f49;
	}

	.social-links {
		display: flex;
		gap: 12px;
		align-items: center;
	}

	.social-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		border-radius: 8px;
		color: #6c765b;
		transition:
			color 0.2s ease,
			transform 0.2s ease,
			background-color 0.2s ease;
	}

	.social-icon:hover {
		color: #4a5d3a;
		background-color: rgba(74, 93, 58, 0.08);
		transform: translateY(-2px);
	}

	.social-icon svg {
		width: 20px;
		height: 20px;
		fill: currentColor;
	}

	section {
		width: calc(min(1024px, 100vw) - 280px - max((1274px - 100vw) / 2, 0px));
		float: right;
		padding-bottom: 50px;
	}

	footer {
		width: 270px;
		float: left;
		position: fixed;
		bottom: min(100vh - 40em, 50px);
		-webkit-font-smoothing: subpixel-antialiased;
	}

	footer p {
		margin-bottom: 5px;
	}

	/* Source code button (used in gallery) */
	:global(.sourceCodeButton) {
		float: right;
		background-color: #fff;
		opacity: 0.9;
		border-radius: 50%;
		margin-top: -50px;
		margin-right: 10px;
		padding: 10px;
		box-shadow: 0 0 4px rgba(0, 0, 0, 0.15);
		-webkit-transform: translateZ(0);
	}

	:global(.sourceCodeButton:hover) {
		opacity: 1;
	}

	:global(.sourceCodeButton img) {
		display: block;
		width: 15px;
		height: 15px;
	}

	@media print, screen and (max-width: 960px) {
		.wrapper {
			width: auto;
			margin: 0;
		}

		header,
		section,
		footer {
			float: none;
			position: static;
			width: auto;
		}

		header {
			padding-right: 320px;
			min-height: auto;
		}

		section {
			border: 1px solid #e5e5e5;
			border-width: 1px 0;
			padding: 20px 0;
			margin: 0 0 20px;
		}

		header :global(ul.nav) {
			position: absolute;
			right: 50px;
			top: 52px;
			display: flex;
			flex-direction: row;
			font-size: min(1.1em, 1.94vw);
			text-align: center;
			align-items: center;
			max-width: calc(100% - 20em);
		}
	}

	@media print, screen and (max-width: 810px) {
		:global(body) {
			word-wrap: break-word;
		}

		header {
			padding: 0;
		}

		header :global(ul.nav),
		header .social-links {
			position: static;
			display: flex;
			max-width: none;
			text-align: left;
			font-size: 1.1em;
		}

		:global(pre, code) {
			word-wrap: normal;
		}
	}

	@media print, screen and (max-width: 480px) {
		:global(body) {
			padding: 15px;
		}
	}

	@media print {
		:global(body) {
			padding: 0.4in;
			font-size: 12pt;
			color: #444;
		}
	}
</style>
