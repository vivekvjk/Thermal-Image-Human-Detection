const { dialog } = require('electron').remote
// const ipcRenderer = require('electron').ipcRenderer
var important
let path = require('path')

const { ipcRenderer } = require('electron')

$(document).ready(() => {
  $('#video-submit').on('click', () => {
    dialog.showOpenDialog({
      filters: [
        { name: 'Movies', extensions: ['mp4,','.avi','*'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    }, (file) => {
      $('#video-submit').text(file)
      // important = file
      console.log('iehfgjr')
      console.log(typeof file)
      ipcRenderer.send('file', file)

    })
    
  })
  $('#webcam-submit').on('click', () => {
    // $('#webcam-submit').text(file)

    // console.log(file)
    ipcRenderer.send('file', null)

  })



})
