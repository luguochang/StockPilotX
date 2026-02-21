import type { ThemeConfig } from 'antd';

export const minimalistTheme: ThemeConfig = {
  token: {
    colorPrimary: '#2563eb',
    borderRadius: 0,
    fontSize: 16,
    fontFamily: '"Sora", "Noto Sans SC", sans-serif',
  },
  components: {
    Card: {
      borderRadiusLG: 0,
      boxShadow: 'none',
      paddingLG: 32,
    },
    Button: {
      borderRadius: 0,
      controlHeight: 48,
      fontSize: 16,
    },
    Table: {
      borderRadius: 0,
      headerBg: '#ffffff',
      headerColor: '#0f172a',
    },
  },
};
