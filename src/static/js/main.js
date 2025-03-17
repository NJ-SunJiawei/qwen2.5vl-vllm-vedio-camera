// 全局对象，供 local.js 和 stream.js 访问
window.stream_sessions = {};

// 菜单切换、图片预览等通用逻辑
document.addEventListener('DOMContentLoaded', () => {
  const menuFile = document.getElementById('menuFile');
  const menuStream = document.getElementById('menuStream');
  const sectionFile = document.getElementById('sectionFile');
  const sectionStream = document.getElementById('sectionStream');

  menuFile.addEventListener('click', (e) => {
    e.preventDefault();
    menuFile.classList.add('active');
    menuStream.classList.remove('active');
    sectionFile.classList.remove('hidden');
    sectionStream.classList.add('hidden');
  });
  menuStream.addEventListener('click', (e) => {
    e.preventDefault();
    menuStream.classList.add('active');
    menuFile.classList.remove('active');
    sectionStream.classList.remove('hidden');
    sectionFile.classList.add('hidden');
  });

  // 图片预览 Modal 逻辑
  let currentModalList = [];
  let currentModalIndex = 0;
  const imageModal = document.getElementById("imageModal");
  const modalImage = document.getElementById("modalImage");
  const modalPrev = document.getElementById("modalPrev");
  const modalNext = document.getElementById("modalNext");

  window.showModal = function(thumbnailList, index) {
    currentModalList = thumbnailList;
    currentModalIndex = index;
    modalImage.src = currentModalList[currentModalIndex];
    imageModal.classList.remove("hidden");
  };

  modalPrev.addEventListener("click", (e) => {
    e.stopPropagation();
    if (currentModalList.length === 0) return;
    currentModalIndex = (currentModalIndex - 1 + currentModalList.length) % currentModalList.length;
    modalImage.src = currentModalList[currentModalIndex];
  });
  
  modalNext.addEventListener("click", (e) => {
    e.stopPropagation();
    if (currentModalList.length === 0) return;
    currentModalIndex = (currentModalIndex + 1) % currentModalList.length;
    modalImage.src = currentModalList[currentModalIndex];
  });
  
  imageModal.addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-nav")) return;
    imageModal.classList.add("hidden");
  });
});

// base64 转 blob
window.base64ToBlob = function(base64Data, contentType = 'image/jpeg') {
  const dataUrl = "data:" + contentType + ";base64," + base64Data;
  return fetch(dataUrl).then(r => r.blob());
};

// 切换全屏
window.toggleFullScreen = function(containerId) {
  const container = document.getElementById(containerId);
  if (!document.fullscreenElement) {
    container.requestFullscreen().catch(err => {
      alert(`全屏错误: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
};
