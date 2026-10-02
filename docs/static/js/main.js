// Page behaviour: math, nav bar, image lightbox, figure tabs, "coming soon" notices, BibTeX copy.
document.addEventListener('DOMContentLoaded', () => {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }],
      throwOnError: false,
    });
  }

  // Nav bar background and scroll-to-top button.
  const nav = document.getElementById('nav');
  const brand = document.getElementById('nav-brand');
  const toTop = document.getElementById('to-top');
  const onScroll = () => {
    const y = window.scrollY;
    nav.classList.toggle('nav-scrolled', y > 40);
    brand.classList.toggle('opacity-0', y <= 40);
    toTop.classList.toggle('hidden', y <= 600);
    toTop.classList.toggle('flex', y > 600);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  toTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

  // Figure tabs: <div data-tabs> with buttons [data-tab=x] and panels [data-panel=x].
  document.querySelectorAll('[data-tabs]').forEach((group) => {
    group.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-tab]');
      if (!btn) return;
      group.querySelectorAll('[data-tab]').forEach((b) => b.classList.toggle('tab-active', b === btn));
      group.querySelectorAll('[data-panel]').forEach((p) => p.classList.toggle('hidden', p.dataset.panel !== btn.dataset.tab));
    });
  });

  // Links that are not public yet show a short notice instead.
  const notice = document.getElementById('notice');
  let noticeTimer = null;
  const showNotice = (text) => {
    notice.querySelector('span').textContent = text;
    notice.classList.add('notice-show');
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => notice.classList.remove('notice-show'), 4000);
  };
  notice.querySelector('button').addEventListener('click', () => notice.classList.remove('notice-show'));

  const lightbox = document.getElementById('lightbox');
  const menu = document.querySelector('.more-menu');
  document.addEventListener('click', (e) => {
    const link = e.target.closest('[data-notice]');
    if (link) {
      e.preventDefault();
      showNotice(link.dataset.notice);
      return;
    }
    const img = e.target.closest('img.zoomable');
    if (img) {
      lightbox.querySelector('img').src = img.src;
      lightbox.classList.replace('hidden', 'flex');
      return;
    }
    if (e.target.closest('#lightbox')) lightbox.classList.replace('flex', 'hidden');

    const moreBtn = e.target.closest('.more-btn');
    menu.classList.toggle('hidden', !moreBtn || !menu.classList.contains('hidden'));
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') lightbox.classList.replace('flex', 'hidden');
  });

  document.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(document.querySelector(btn.dataset.copy).textContent);
      btn.innerHTML = '<i class="fa-regular fa-circle-check"></i><span>Copied</span>';
      setTimeout(() => (btn.innerHTML = '<i class="fa-regular fa-copy"></i><span>Copy</span>'), 1500);
    });
  });
});
