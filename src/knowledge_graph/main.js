import { createApp } from 'vue';
import App from './App.vue';
import router from './router';

// 如果你有全局样式，可以在这里导入
import './assets/styles.css'; // 假设你的全局样式文件是styles.css

// 如果你有状态管理工具（例如Vuex），可以在这里导入并使用
// import store from './store'; 

// 创建Vue应用实例
const app = createApp(App);

// 使用路由和状态管理
app.use(router);
// app.use(store); // 如果你使用了Vuex或其他状态管理工具

// 挂载应用到DOM
app.mount('#app');

import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import store from './store'; // 导入Vuex store

import './assets/styles.css'; // 导入全局样式

const app = createApp(App);

app.use(router);
app.use(store); // 使用Vuex store

app.mount('#app');
