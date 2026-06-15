import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI4EDA 知识库',
  description: '深度学习集成电路设计专题',
  lang: 'zh-CN',
  themeConfig: {
    siteTitle: 'AI4EDA',
    nav: [
      { text: '首页', link: '/' },
      {
        text: '课程内容',
        items: [
          { text: '第一部分：EDA与Python', link: '/part1/' },
          { text: '第二部分：深度学习与TCAD', link: '/part2/' },
          { text: '第三部分：APR Placement', link: '/part3/' },
          { text: '第四部分：APR CTS', link: '/part4/' },
          { text: '第五部分：Route与综合', link: '/part5/' },
        ]
      }
    ],
    sidebar: [
      {
        text: '第一部分：EDA集成电路设计自动化和Python编程',
        collapsed: false,
        items: [
          { text: '概述', link: '/part1/' },
          { text: '1.1 EDA设计自动化基本概念', link: '/part1/eda-basics' },
          { text: '1.2 芯片设计流程与EDA应用', link: '/part1/chip-flow' },
          { text: '1.3 Python编程基础', link: '/part1/python-basics' },
          { text: '1.4 超图分割算法', link: '/part1/hypergraph' },
          { text: '1.5 上机实验', link: '/part1/lab' },
        ]
      },
      {
        text: '第二部分：深度学习方法、Pytorch与TCAD仿真加速',
        collapsed: false,
        items: [
          { text: '概述', link: '/part2/' },
          { text: '2.1 深度学习基本理论', link: '/part2/dl-theory' },
          { text: '2.2 深度学习发展历程', link: '/part2/dl-history' },
          { text: '2.3 TCAD仿真与器件建模', link: '/part2/tcad' },
          { text: '2.4 上机实验', link: '/part2/lab' },
        ]
      },
      {
        text: '第三部分：APR Placement布局算法',
        collapsed: false,
        items: [
          { text: '概述', link: '/part3/' },
          { text: '3.1 后端物理设计流程', link: '/part3/pd-flow' },
          { text: '3.2 Placement过程详解', link: '/part3/placement-detail' },
          { text: '3.3 Placement相关算法', link: '/part3/algorithms' },
          { text: '3.4 图卷积神经网络', link: '/part3/gcn' },
          { text: '3.5 强化学习基础', link: '/part3/rl' },
          { text: '3.6 深度学习用于Placement优化', link: '/part3/dl-placement' },
          { text: '3.7 上机实验', link: '/part3/lab' },
        ]
      },
      {
        text: '第四部分：APR CTS时钟树综合',
        collapsed: false,
        items: [
          { text: '概述', link: '/part4/' },
          { text: '4.1 时钟树基本概念', link: '/part4/clock-basics' },
          { text: '4.2 CTS过程详解', link: '/part4/cts-detail' },
          { text: '4.3 CTS相关算法', link: '/part4/algorithms' },
          { text: '4.4 深度学习与CTS', link: '/part4/dl-cts' },
          { text: '4.5 上机实验', link: '/part4/lab' },
        ]
      },
      {
        text: '第五部分：Route、逻辑综合与电源完整性',
        collapsed: false,
        items: [
          { text: '概述', link: '/part5/' },
          { text: '5.1 芯片绕线Routing过程', link: '/part5/routing-detail' },
          { text: '5.2 Routing相关算法', link: '/part5/algorithms' },
          { text: '5.3 深度学习与Routing', link: '/part5/dl-routing' },
          { text: '5.4 IR Drop预测', link: '/part5/ir-drop' },
          { text: '5.5 逻辑综合与深度学习', link: '/part5/logic-synthesis' },
          { text: '5.6 Transformer模型', link: '/part5/transformer' },
          { text: '5.7 上机实验', link: '/part5/lab' },
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/' }
    ],
    footer: {
      message: 'AI4EDA 深度学习集成电路设计专题知识库',
      copyright: '© 2025 AI4EDA'
    },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '未找到结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },
    outline: {
      label: '页面导航',
      level: [2, 3]
    },
    lastUpdated: {
      text: '最后更新'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    }
  },
  markdown: {
    lineNumbers: true
  },
  lastUpdated: true
})
