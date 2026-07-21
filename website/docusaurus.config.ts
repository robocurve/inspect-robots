import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const githubDocsEditUrl =
  'https://github.com/robocurve/inspect-robots/edit/main/docs/';

const config: Config = {
  title: 'Inspect Robots',
  tagline:
    'An open-source evaluation framework for benchmarking AI and robots in the physical world',
  favicon: 'img/favicon.svg',
  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'icon',
        type: 'image/png',
        href: '/img/favicon.png',
      },
    },
  ],
  url: 'https://inspectrobots.org',
  baseUrl: '/',
  organizationName: 'robocurve',
  projectName: 'inspect-robots',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: ({docPath}) =>
            docPath === 'api/index.md'
              ? undefined
              : `${githubDocsEditUrl}${docPath}`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  plugins: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        docsRouteBasePath: '/',
        docsDir: '../docs',
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],
  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Inspect Robots',
      logo: {
        alt: 'Inspect Robots logo',
        src: 'img/inspect-robots-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/robocurve/inspect-robots',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'Copyright © Inspect Robots contributors',
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
