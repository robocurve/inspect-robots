import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Guide',
      collapsed: false,
      items: [
        'guide/quickstart',
        'guide/concepts',
        'guide/writing-a-benchmark',
        'guide/policies-and-embodiments',
        'guide/scoring',
        'guide/logging-and-rerun',
        'guide/plugins',
        'guide/adapters',
        'guide/cli',
      ],
    },
    {
      type: 'category',
      label: 'Cookbooks',
      collapsed: false,
      items: ['cookbooks/gr00t-on-yam'],
    },
    {
      type: 'doc',
      id: 'api/index',
      label: 'API reference',
    },
  ],
};

export default sidebars;
