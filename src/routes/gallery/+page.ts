import { galleryItems } from '$lib/data/gallery';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ url }) => {
	const tab = url.searchParams.get('tab');
	const validTab = tab && galleryItems.some((i) => i.id === tab) ? tab : galleryItems[0].id;
	return { tab: validTab };
};
