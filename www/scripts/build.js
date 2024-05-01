import fs from 'fs'
import path from 'path'
const b = async () => {
  const sourceDir = 'docs'
  const {
    loadUserConfig,
    resolveAppConfig,
    resolveCliAppConfig,
    resolveUserConfigConventionalPath,
    transformUserConfigToPlugin
  } = await import('vuepress/cli')

  const { createBuildApp } = await import('vuepress/core')
  console.log('createBuildApp')

  const cliAppConfig = resolveCliAppConfig(sourceDir, {})

  // resolve user config file
  const userConfigPath = resolveUserConfigConventionalPath(cliAppConfig.source)

  const { userConfig } = await loadUserConfig(userConfigPath)

  // resolve the final app config to use
  const appConfig = resolveAppConfig({
    defaultAppConfig: {},
    cliAppConfig,
    userConfig
  })

  if (appConfig === null) return

  const staticSiteApp = createBuildApp(appConfig)

  // use user-config plugin
  staticSiteApp.use(
    transformUserConfigToPlugin(userConfig, cliAppConfig.source)
  )

  // initialize and prepare
  console.log('staticSiteApp.init')
  await staticSiteApp.init()
  console.log('staticSiteApp.prepare')
  await staticSiteApp.prepare()

  // build
  // TODO: update percent on build process
  console.log('staticSiteApp.build')
  await staticSiteApp.build()

  // process onGenerated hook
  console.log('staticSiteApp.pluginApi.hooks.onGenerated.process')
  await staticSiteApp.pluginApi.hooks.onGenerated.process(staticSiteApp)

  const currentDirectory = process.cwd()
  const parentDirectory = path.resolve(currentDirectory, '')
  let json = path.join(parentDirectory, 'docs/data.json')
  const data = fs.readFileSync(json, 'utf8')
  let myConfig = JSON.parse(data)

  fs.writeFile(
    path.join(path.resolve(myConfig.dest, ''), 'CNAME'),
    myConfig.hostname,
    err => {
      if (err) {
        console.error('写入文件时出错：', err)
      } else {
        console.log('成功将内容写入CNAME文件')
      }
    }
  )
}

b()
