
// const { ipcMain } = require('electron')
const { app, BrowserWindow, Menu, dialog } = require('electron')
const path = require('path')
const { ipcMain } = require('electron')
const { exec } = require('child_process')

ipcMain.on('file', (event, arg) => {
  var command
  if(arg){
    command = `cd && cd ${path.join(__dirname, '../../object_detection/thermal_detect/object_detection')} && python3 test.py ${arg.length>0 ? `\"${arg[0]}\"` : ""}`
  }else{
    command = `cd && cd ${path.join(__dirname, '../../object_detection/thermal_detect/object_detection')} && python3 test.py`

  }
  console.log(command)
  exec(command, (err, stdout, stderr) => {
    if (err) {
    // node couldn't execute the command
      return
    }
    // the *entire* stdout and stderr (buffered)
    console.log(`stdout: ${stdout}`)
    console.log(`stderr: ${stderr}`)
  })
})

let window

function createWindow () {
  window = new BrowserWindow({
    width: 800,
    height: 600,
    show: true,
    frame: true,
    fullscreenable: false,
    resizable: false,
    transparent: false,
    hasShadow: true,
    webPreferences: {
      backgroundThrottling: false,
      nodeIntegration: true
    }
  })
  window.openDevTools()

  window.setMenu(null)

  if (process.platform === 'darwin') {
    // Create our menu entries so that we can use MAC shortcuts
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      {
        label: 'Edit',
        submenu: [
          { role: 'quit' },
          { role: 'cut' },
          { role: 'copy' },
          { role: 'paste' },
          { role: 'selectall' }
        ]
      }
    ]))
  }

  window.loadURL(`file://${path.join(__dirname, './views/index.html')}`)
  window.on('closed', function () {
    window = null
  })
}

app.on('ready', createWindow)

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', function () {
  if (window === null) createWindow()
})
