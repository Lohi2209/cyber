document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('.analysis-form');
    const textarea = document.querySelector('#message');
    const charCount = document.querySelector('#char-count');

    const updateCounter = () => {
        if (textarea && charCount) {
            charCount.textContent = `${textarea.value.length} characters`;
        }
    };

    if (textarea) {
        textarea.addEventListener('input', updateCounter);
        updateCounter();
    }

    document.querySelectorAll('[data-sample]').forEach((button) => {
        button.addEventListener('click', () => {
            if (!textarea) {
                return;
            }

            textarea.value = button.dataset.sample || '';
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
            textarea.focus();
        });
    });

    if (form) {
        form.addEventListener('submit', () => {
            const button = form.querySelector('button[type="submit"]');
            if (button) {
                button.textContent = 'Analyzing...';
                button.disabled = true;
            }
        });
    }
});
