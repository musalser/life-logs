<script setup>
// «Дневник» слит с «Рукописями» в единую вкладку «Записи». Старые ссылки
// (/diary?page=N) ведут на ту же страницу в новом дереве.
const route = useRoute()
const query = route.query.page ? { type: 'diary', page: route.query.page } : { type: 'diary' }
await navigateTo({ path: '/library', query }, { redirectCode: 301 })
</script>
