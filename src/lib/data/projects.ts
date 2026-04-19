export interface Project {
	title: string;
	type: string;
	description: string;
	image: string;
	link: string;
	tags: string[];
	time: string;
	authors: string;
	spanFull?: boolean;
}

export const projects: Project[] = [
	{
		title: 'Manga Text Representation',
		type: 'Research',
		description:
			'Improving image representations of multilingual text for manga understanding and region-specific color transfer.',
		image: '/img/manga_logo.png',
		link: 'https://seahhorse.github.io/projects/improving_image_representations_of_words_through_region-specific_loss_minimisation/',
		tags: ['Python', 'PyTorch', 'NLP', 'Computer Vision'],
		time: 'May 2024',
		authors: 'Ming Chong Lim, Shao Xuan Seah'
	},
	{
		title: 'Sketch2Image',
		type: 'Research',
		description:
			'Image synthesis from freehand sketches using deep generative models and latent variable manipulation.',
		image: '/img/imgsyn_proj5_logo.png',
		link: 'https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj5/',
		tags: ['Python', 'GANs', 'Image Synthesis'],
		time: 'Apr 2024',
		authors: 'Ming Chong Lim'
	},
	{
		title: 'Neural Style Transfer',
		type: 'Research',
		description:
			'Artistic style transfer using neural networks for texture synthesis and real-time image transformation.',
		image: '/img/imgsyn_proj4_logo.png',
		link: 'https://www.andrew.cmu.edu/course/16-726-sp24/projects/mingchol/proj4/',
		tags: ['Deep Learning', 'Style Transfer'],
		time: 'Mar 2024',
		authors: 'Ming Chong Lim',
		spanFull: true
	},
	{
		title: 'RouteMaker',
		type: 'Application',
		description:
			'A cross-platform application utilizing machine learning to allow users to quickly create climbing routes and share them with the community.',
		image: '/img/routemaker_logo.jpg',
		link: 'https://github.com/nandium/RouteMaker',
		tags: ['Mobile', 'ML', 'Cross-platform'],
		time: 'Apr 2021 – Sep 2023',
		authors: 'Ming Chong Lim, Yar Khine Phyo'
	},
	{
		title: 'Travelling BusinessMan',
		type: 'Application',
		description:
			'A brownfield project done in Java under the module CS2103T Software Engineering.',
		image: '/img/tbm_logo.png',
		link: 'https://github.com/AY2021S1-CS2103T-F11-4/tp',
		tags: ['Java', 'Software Engineering'],
		time: 'Aug 2020 – Nov 2020',
		authors: 'Ming Chong Lim'
	}
];
