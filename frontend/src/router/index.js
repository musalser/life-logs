import { createRouter, createWebHistory } from 'vue-router';

import ChatPage from '../pages/ChatPage.vue';
import DiaryPage from '../pages/DiaryPage.vue';
import ListPage from '../pages/ListPage.vue';

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', name: 'chat', component: ChatPage },
  { path: '/diary', name: 'diary', component: DiaryPage },
  { path: '/list', name: 'list', component: ListPage }
];

export default createRouter({
  history: createWebHistory(),
  routes
});
