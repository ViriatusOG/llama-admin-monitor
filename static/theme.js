// Theme system: THEMES registry (single source of truth for shipped themes),
// persisted selection, root data-theme attribute, color-scheme hint, and the
// sidebar theme menu.
//
// Adding a theme is one entry in THEMES plus one token block in tokens.css.
(function () {
    'use strict';

    const THEME_STORAGE_KEY = 'llama_admin_monitor_theme';
    // The pre-paint script in index.html cannot see THEMES (it must stay
    // inline and blocking). It reads the resolved scheme from here instead, so
    // a stored light theme does not flash the dark default while this loads.
    const THEME_SCHEME_STORAGE_KEY = 'llama_admin_monitor_theme_scheme';

    // `scheme` drives the color-scheme hint and must match the token block's
    // own `color-scheme` declaration, or native controls and scrollbars get
    // styled for the opposite polarity.
    const THEMES = [
        { id: 'tokyo', label: 'Tokyo', hint: 'Dark', scheme: 'dark', swatchBg: '#161824', swatchAccent: '#6c9bff' },
        { id: 'nebula', label: 'Nebula', hint: 'Dark', scheme: 'dark', swatchBg: '#121420', swatchAccent: '#8b5cf6' },
        { id: 'graphite', label: 'Graphite', hint: 'Mid', scheme: 'dark', swatchBg: '#383b41', swatchAccent: '#d9a05b' },
        { id: 'cappuccino', label: 'Cappuccino', hint: 'Light', scheme: 'light', swatchBg: '#fff4e6', swatchAccent: '#4b3832' },
        { id: 'mint', label: 'Mint', hint: 'Light', scheme: 'light', swatchBg: '#e3f0e9', swatchAccent: '#276947' },
    ];

    const DEFAULT_THEME = THEMES[0].id;
    const THEMES_BY_ID = new Map(THEMES.map(t => [t.id, t]));

    function normalizeTheme(theme) {
        return THEMES_BY_ID.has(theme) ? theme : DEFAULT_THEME;
    }

    function getStoredTheme() {
        try {
            return normalizeTheme(localStorage.getItem(THEME_STORAGE_KEY));
        } catch (_) {
            return DEFAULT_THEME;
        }
    }

    function updateColorScheme(theme) {
        const scheme = THEMES_BY_ID.get(theme).scheme;
        try { localStorage.setItem(THEME_SCHEME_STORAGE_KEY, scheme); } catch (_) {}
        const meta = document.querySelector('meta[name="color-scheme"]');
        if (meta) meta.setAttribute('content', scheme === 'light' ? 'light dark' : 'dark light');
        // The PWA chrome colour follows the theme's surface tone.
        const themeColor = document.querySelector('meta[name="theme-color"]');
        if (themeColor) themeColor.setAttribute('content', THEMES_BY_ID.get(theme).swatchBg);
    }

    function applyTheme(theme, options = {}) {
        const next = normalizeTheme(theme);
        // Always set the attribute, including for the default theme, so every
        // theme is selected the same way and none is "the absent case".
        document.documentElement.dataset.theme = next;
        updateColorScheme(next);
        if (options.persist) {
            try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch (_) {}
        }
        refreshSwitcher(next);
        return next;
    }

    function getCurrentTheme() {
        return normalizeTheme(document.documentElement.dataset.theme || DEFAULT_THEME);
    }

    function swatchGradient(theme) {
        return 'linear-gradient(135deg, ' + theme.swatchBg + ' 55%, ' + theme.swatchAccent + ' 45%)';
    }

    function refreshSwitcher(active = getCurrentTheme()) {
        document.querySelectorAll('[data-theme-option]').forEach(button => {
            const isActive = normalizeTheme(button.dataset.themeOption) === active;
            button.classList.toggle('active', isActive);
            button.setAttribute('aria-checked', String(isActive));
            button.tabIndex = isActive ? 0 : -1;
        });
        const theme = THEMES_BY_ID.get(active);
        const label = document.getElementById('theme-menu-current');
        if (label && theme) label.textContent = theme.label;
        const swatch = document.getElementById('theme-menu-current-swatch');
        if (swatch && theme) swatch.style.background = swatchGradient(theme);
    }

    const CHECK_ICON = '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>';

    function renderMenu() {
        const list = document.getElementById('theme-menu-list');
        if (!list) return [];
        list.textContent = '';
        for (const theme of THEMES) {
            const item = document.createElement('li');
            item.setAttribute('role', 'none');

            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'theme-menu-item';
            button.setAttribute('role', 'menuitemradio');
            button.setAttribute('aria-checked', 'false');
            button.dataset.themeOption = theme.id;
            button.tabIndex = -1;

            const swatch = document.createElement('span');
            swatch.className = 'theme-menu-swatch';
            swatch.setAttribute('aria-hidden', 'true');
            swatch.style.background = swatchGradient(theme);

            const label = document.createElement('span');
            label.className = 'theme-menu-label';
            label.textContent = theme.label;

            const hint = document.createElement('span');
            hint.className = 'theme-menu-hint';
            hint.textContent = theme.hint;

            const check = document.createElement('span');
            check.className = 'icon icon-sm theme-menu-check';
            check.setAttribute('aria-hidden', 'true');
            check.innerHTML = CHECK_ICON;

            button.append(swatch, label, hint, check);
            item.appendChild(button);
            list.appendChild(item);
        }
        return Array.from(list.querySelectorAll('[data-theme-option]'));
    }

    function initMenu(items) {
        const trigger = document.getElementById('theme-menu-trigger');
        const list = document.getElementById('theme-menu-list');
        if (!trigger || !list) return;

        const isOpen = () => trigger.getAttribute('aria-expanded') === 'true';

        function setOpen(open, focusItem = true) {
            list.hidden = !open;
            trigger.setAttribute('aria-expanded', String(open));
            if (open && focusItem) {
                const checked = items.find(i => i.getAttribute('aria-checked') === 'true');
                (checked || items[0]).focus();
            }
        }

        function moveFocus(from, delta) {
            if (!items.length) return;
            const index = items.indexOf(from);
            items[(index + delta + items.length) % items.length].focus();
        }

        trigger.addEventListener('click', () => setOpen(!isOpen()));
        trigger.addEventListener('keydown', e => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                if (!isOpen()) setOpen(true);
                else items[e.key === 'ArrowDown' ? 0 : items.length - 1].focus();
            }
        });

        for (const item of items) {
            item.addEventListener('click', () => {
                applyTheme(item.dataset.themeOption, { persist: true });
                setOpen(false, false);
                trigger.focus();
            });
            item.addEventListener('keydown', e => {
                switch (e.key) {
                    case 'ArrowDown': e.preventDefault(); moveFocus(item, 1); break;
                    case 'ArrowUp': e.preventDefault(); moveFocus(item, -1); break;
                    case 'Home': e.preventDefault(); items[0].focus(); break;
                    case 'End': e.preventDefault(); items[items.length - 1].focus(); break;
                    case 'Escape': e.preventDefault(); setOpen(false, false); trigger.focus(); break;
                    case 'Tab': setOpen(false, false); break;
                    default: break;
                }
            });
        }

        document.addEventListener('click', e => {
            if (!isOpen()) return;
            if (e.target && typeof e.target.closest === 'function' && e.target.closest('.theme-menu')) return;
            setOpen(false, false);
        });
    }

    function init() {
        const rendered = renderMenu();
        applyTheme(getStoredTheme());
        initMenu(rendered);
        refreshSwitcher();
    }

    window.LlamaAdmin = window.LlamaAdmin || {};
    window.LlamaAdmin.theme = { applyTheme, getCurrentTheme, init, THEMES, storageKey: THEME_STORAGE_KEY };
})();
