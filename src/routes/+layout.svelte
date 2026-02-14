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
.main-title {
	padding-left: 17.0rem;
}
</style>

<div class="container-fluid">
	<a href="/" class="text-decoration-none text-dark">
  <h4 class="heading-title main-title pt-3">Svanik Sharma's Website</h4>
	</a>
</div>

{@render children()}
