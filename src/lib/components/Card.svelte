<script lang="ts">
	interface Props {
		link: string;
		linktarget?: string;
		image: string;
		alt: string;
		title: string;
		description: string;
		time: string;
		authors: string;
		tags?: string[];
	}

	let {
		link,
		linktarget = '_blank',
		image,
		alt,
		title,
		description,
		time,
		authors,
		tags = []
	}: Props = $props();
</script>

<article class="card">
	<figure class="card-figure">
		<img class="card-image" src={image} {alt} />
	</figure>
	{#if tags.length > 0}
		<div class="card-tags">
			{#each tags as tag}
				<span class="card-tag">{tag}</span>
			{/each}
		</div>
	{/if}
	<header class="card-title">{title}</header>
	<div class="card-desc">
		{description}
	</div>
	<footer class="card-footer">
		<small>{time} | {authors}</small>
	</footer>
	<a class="card-link" href={link} target={linktarget} title="[Opens in a new tab] {title}"></a>
</article>

<style>
	.card {
		background-color: #f2f1f7;
		position: relative;
		display: flex;
		flex-direction: column;
		width: 100%;
		height: auto;
		padding: 1.2em;
		border-radius: 0.85em;
		box-shadow: 4px 4px rgba(0, 0, 0, 0.1);
		transition:
			transform 0.25s ease,
			box-shadow 0.25s ease,
			background-color 0.25s ease;
	}

	.card:hover {
		background-color: #ececf3;
		transform: translateY(-3px);
		box-shadow: 4px 8px 16px rgba(0, 0, 0, 0.1);
	}

	.card-figure {
		width: 100%;
		margin: 0;
	}

	.card-figure img {
		border-radius: 0.7em;
		width: 100%;
		aspect-ratio: 16 / 10;
		object-fit: cover;
	}

	.card-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-top: 0.7em;
	}

	.card-tag {
		font-size: 0.7em;
		font-weight: 600;
		letter-spacing: 0.03em;
		padding: 3px 9px;
		border-radius: 4px;
		background: rgba(74, 93, 58, 0.1);
		color: #4a5d3a;
	}

	.card-title {
		position: static;
		width: auto;
		display: block;
		min-height: auto;
		font-size: 1.25em;
		font-weight: 600;
		margin: 0.5em 0 0.3em;
		line-height: 1.3;
	}

	.card-desc {
		font-size: 0.95em;
		margin-bottom: 0.4em;
		color: #8b8b8b;
		line-height: 1.5;
		flex: 1;
	}

	.card-footer {
		position: static;
		width: auto;
		color: #8b8b8b;
		margin-top: auto;
	}

	.card-link {
		position: absolute;
		z-index: 999999;
		display: block;
		width: 100%;
		height: 100%;
		top: 0;
		left: 0;
	}

	.card-link::after {
		display: none;
	}
</style>
