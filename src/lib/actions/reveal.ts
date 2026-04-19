/**
 * Svelte action that adds a 'visible' class when the element enters the viewport.
 * Used for scroll-triggered animations throughout the site.
 */
export function reveal(
	node: HTMLElement,
	options?: { threshold?: number; rootMargin?: string; once?: boolean }
) {
	const { threshold = 0.12, rootMargin = '0px 0px -40px 0px', once = true } = options ?? {};

	const observer = new IntersectionObserver(
		([entry]) => {
			if (entry.isIntersecting) {
				node.classList.add('visible');
				if (once) observer.unobserve(node);
			}
		},
		{ threshold, rootMargin }
	);

	observer.observe(node);

	return {
		destroy() {
			observer.disconnect();
		}
	};
}

/**
 * Svelte action for shutter line animation — opens when element enters viewport.
 */
export function shutter(node: HTMLElement) {
	const observer = new IntersectionObserver(
		([entry]) => {
			if (entry.isIntersecting) {
				node.classList.add('open');
				observer.unobserve(node);
			}
		},
		{ threshold: 0.5 }
	);

	observer.observe(node);

	return {
		destroy() {
			observer.disconnect();
		}
	};
}
