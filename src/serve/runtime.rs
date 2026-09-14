use anyhow::{Context, Result};
use std::{
    io::Write,
    os::unix::fs::{FileTypeExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};
use tokio::net::{UnixListener, UnixStream};

pub(super) struct RuntimeFiles {
    socket: Option<PathBuf>,
    pid: PathBuf,
}

impl RuntimeFiles {
    pub async fn bind(socket: &Path, pid: &Path) -> Result<(Self, UnixListener)> {
        for path in [socket, pid] {
            std::fs::create_dir_all(path.parent().context("Runtime path has no parent")?)?;
        }
        let mut pid_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(pid)?;
        writeln!(pid_file, "{}", std::process::id())?;
        let mut files = Self {
            socket: None,
            pid: pid.to_owned(),
        };
        if socket.exists() {
            anyhow::ensure!(
                std::fs::symlink_metadata(socket)?.file_type().is_socket(),
                "Refusing to replace a non-socket file"
            );
            anyhow::ensure!(
                UnixStream::connect(socket).await.is_err(),
                "Another engine is using this socket"
            );
            std::fs::remove_file(socket)?;
        }
        let listener = UnixListener::bind(socket)?;
        files.socket = Some(socket.to_owned());
        std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
        Ok((files, listener))
    }
}

impl Drop for RuntimeFiles {
    fn drop(&mut self) {
        if let Some(socket) = &self.socket {
            let _ = std::fs::remove_file(socket);
        }
        let _ = std::fs::remove_file(&self.pid);
    }
}
