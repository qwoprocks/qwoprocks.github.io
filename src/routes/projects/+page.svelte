<script lang="ts">
	import Card from '$lib/components/Card.svelte';
</script>

<svelte:head>
	<title>Projects | Ming Chong Lim</title>
</svelte:head>

<div class="project-grid">
	<Card
		link="https://seahhorse.github.io/projects/improving_image_representations_of_words_through_region-specific_loss_minimisation/"
		image="/img/manga_logo.png"
		alt="A sample manga panel from the Manga109 dataset"
		title="Improving Image Representations of Words through Region-specific Loss Minimisation"
		description="How can we make text in generated images seem more real?"
		time="22 May 2024"
		authors="Ming Chong Lim, Shao Xuan Seah"
		tags={['NLP', 'Computer Vision']}
	/>

	<Card
		link="https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj5/"
		image="/img/imgsyn_proj5_logo.png"
		alt="Logo for sketch2image"
		title="Sketch2Image"
		description="A project done for 16-726 Image Synthesis at CMU involving image synthesis through latent variable manipulation"
		time="Apr 2024"
		authors="Ming Chong Lim"
		tags={['Image Synthesis', 'GANs']}
	/>

	<Card
		link="https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj4/"
		image="/img/imgsyn_proj4_logo.png"
		alt="Logo for neural style transfer"
		title="Neural Style Transfer"
		description="A project done for 16-726 Image Synthesis at CMU involving texture synthesis and style transfer"
		time="Mar 2024"
		authors="Ming Chong Lim"
		tags={['Style Transfer', 'Deep Learning']}
	/>

	<Card
		link="https://github.com/nandium/RouteMaker"
		image="/img/routemaker_logo.jpg"
		alt="Logo for RouteMaker"
		title="RouteMaker (Front-end Developer)"
		description="A cross-platform application utilizing machine learning to allow users to quickly create climbing routes and share them with the community"
		time="Apr 2021 - Sep 2023"
		authors="Ming Chong Lim, Yar Khine Phyo"
		tags={['Mobile', 'ML', 'Cross-platform']}
	/>

	<Card
		link="https://github.com/AY2021S1-CS2103T-F11-4/tp"
		image="/img/tbm_logo.png"
		alt="Logo for Travelling BusinessMan"
		title="Travelling BusinessMan (Software Developer)"
		description="A brownfield project done in Java under the module CS2103T Software Engineering"
		time="Aug 2020 - Nov 2020"
		authors="Ming Chong Lim"
		tags={['Java', 'Software Engineering']}
	/>
</div>

<style>
	.project-grid {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 1.5em;
	}

	@media screen and (max-width: 640px) {
		.project-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
