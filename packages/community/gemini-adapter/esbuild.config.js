const esbuildPluginTsc = require('esbuild-plugin-tsc');

const external = ['@nestjs/*', '@vm-x-ai/completion-provider', '@grpc/grpc-js', 'nestjs-otel', 'rxjs'];

/**
 * @type {import('esbuild').BuildOptions}
 */
module.exports = {
  bundle: true,
  platform: 'node',
  target: ['node20'],
  format: 'cjs',
  external,
  minify: false,
  keepNames: true,
  footer: {
    js: 'module.exports = src_exports;',
  },
  plugins: [
    esbuildPluginTsc({
      tsconfigPath: `${__dirname}/tsconfig.lib.json`,
    }),
  ],
};
