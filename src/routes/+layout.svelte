<script lang="ts">
	let { children } = $props();
</script>

<svelte:head>
	<title>Svanik Sharma's Website</title>
	<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB" crossorigin="anonymous">
</svelte:head>

<style>
:root {
  --bs-body-font-family: 'Ubuntu', system-ui, -apple-system, sans-serif;
  --bs-font-sans-serif: 'Ubuntu', system-ui, -apple-system, sans-serif;
}
.heading-title {
	font-weight: 700;
}
</style>

<header class="text-center pt-3">
	<a href="/" class="text-decoration-none text-dark">
		<h4 class="heading-title">Svanik Sharma's Website</h4>
	</a>
</header>

{@render children()}
