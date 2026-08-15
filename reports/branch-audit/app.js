const railLinks = [...document.querySelectorAll('.rail > a')];
const sections = [...document.querySelectorAll('main > section[id]')];

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    railLinks.forEach((link) => {
      link.classList.toggle('active', link.getAttribute('href') === `#${visible.target.id}`);
    });
  }, { rootMargin: '-18% 0px -64% 0px', threshold: [0.01, 0.2, 0.5] });
  sections.forEach((section) => observer.observe(section));
}

const search = document.querySelector('#reference-search');
const references = [...document.querySelectorAll('#reference-list article')];
const noResults = document.querySelector('#no-results');

search?.addEventListener('input', () => {
  const query = search.value.trim().toLowerCase();
  let shown = 0;
  references.forEach((reference) => {
    const haystack = `${reference.dataset.search || ''} ${reference.textContent}`.toLowerCase();
    const matches = !query || haystack.includes(query);
    reference.hidden = !matches;
    if (matches) shown += 1;
  });
  noResults.hidden = shown !== 0;
});

const copyButton = document.querySelector('#copy-link');
const toast = document.querySelector('#toast');

copyButton?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(window.location.href);
    toast.textContent = 'Link copied';
  } catch {
    toast.textContent = 'Copy unavailable';
  }
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 1800);
});
