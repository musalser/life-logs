<script setup>
// «Рукописи» слиты с «Дневником» в единую вкладку «Записи». Старые ссылки
// (/manuscripts?page=N) ведут на ту же страницу в новом дереве.
const route = useRoute()
const query = route.query.page ? { type: 'htr', page: route.query.page } : { type: 'htr' }
await navigateTo({ path: '/library', query }, { redirectCode: 301 })
</script>
