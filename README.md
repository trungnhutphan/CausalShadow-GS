# CausalShadow-GS

Thí nghiệm trên cảnh `lego` của NeRF-Synthetic nhằm kiểm tra quy trình từ ảnh đa góc nhìn và camera đã biết (khối 1), qua tối ưu 3D Gaussian Splatting của GraphDECO, đến tập Gaussian biểu diễn cảnh (khối 2). Giai đoạn P00–P06 bàn giao dữ liệu đầu vào cho bước phân tích ảnh và camera. Tái tạo, render, đánh giá và các khối 3–13 thuộc những giai đoạn sau. [Bài báo NeRF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1473_ECCV_2020_paper.php) · [3D Gaussian Splatting của GraphDECO](https://github.com/graphdeco-inria/gaussian-splatting)

## Bàn giao ngày 23/11/2026

Lần chuẩn bị dữ liệu mang mã `p00-p06_001`. Các số liệu dưới đây được đo trên bản đã giải nén. [Manifest](runs/lego/p00-p06_001/manifest.json) · [SHA-256 cho từng tệp](runs/lego/p00-p06_001/SHA256SUMS)

| Tập | Frame có ảnh màu và pose | Ảnh phụ trợ |
| --- | ---: | --- |
| train | 100 | 0 |
| val | 100 | 0 |
| test | 200 | 200 depth và 200 normal |

Cả 400 ảnh màu là PNG RGBA 800 × 800. Mỗi frame trong ba tệp `transforms_*.json` trỏ đến một ảnh màu trong tập tương ứng. `transform_matrix` biểu diễn camera-to-world theo quy ước Blender/OpenGL. Ba tệp khai báo cùng góc nhìn ngang 0.6911112070083618 radian. [Quy ước camera của tác giả](https://github.com/bmild/nerf#already-have-poses)

## Tổ chức thư mục

| Đường dẫn | Vai trò |
| --- | --- |
| `data/raw/nerf_synthetic/lego/` | Ảnh và JSON gốc từ tác giả. Các kiểm tra bàn giao chỉ đọc thư mục này. |
| `src/causalshadow/` | Package Python của dự án. Hai module `inspect_dataset.py` và `inspect_gaussians.py` hiện là khung cho các giai đoạn phân tích tương ứng. |
| `tests/test_lego_handoff.py` | Kiểm tra số frame, đường dẫn, thuộc tính ảnh, ma trận pose và ảnh depth/normal trên bản Lego đã tải. |
| `third_party/gaussian-splatting/` | Dành cho mã GraphDECO khi chuẩn bị reconstruction. |
| `runs/lego/<run_id>/input/` | Dữ liệu làm việc của lần chạy. Khi dùng GraphDECO, thư mục này cho phép trình đọc tạo thêm `points3d.ply` bên ngoài dữ liệu gốc. |
| `runs/lego/<run_id>/model/` | Đích lưu Gaussian PLY, checkpoint và ảnh render ở giai đoạn tái tạo. |
| `runs/lego/<run_id>/manifest.json` | Mã lần chạy, nguồn, cấu hình, các phép kiểm tra và trạng thái bàn giao. |

## Kiểm tra lại bản bàn giao

Chạy tại gốc `CausalShadow-GS/` với Python 3.10 trở lên, NumPy và Pillow. Máy đã thực hiện bàn giao dùng Python 3.10.11, NumPy 2.2.6 và Pillow 12.0.0. Trên môi trường Python mới, lệnh `python3 -m pip install -e '.[inspection]'` cài package và các phụ thuộc phục vụ kiểm tra.

```sh
python3 -B -m unittest discover -s tests -v
shasum -a 256 runs/lego/p00-p06_001/SHA256SUMS
(
  cd data/raw/nerf_synthetic/lego &&
  shasum -a 256 -c ../../../../runs/lego/p00-p06_001/SHA256SUMS
)
```

Hai kiểm thử phải báo `OK`. SHA-256 của chính bảng checksum là `80268d4c533fa39ca30f03f7cf9c029847d1f99d4e1a32d7f276797057692114`. Lệnh cuối kiểm tra từng tệp trong 803 tệp bàn giao. Tất cả mã băm trong bàn giao được tính cục bộ.

## Khôi phục dữ liệu trong một bản dự án mới

Script của tác giả trỏ tới gói ví dụ UCSD chứa cảnh Lego và Fern. Phiên bàn giao dùng HTTPS và trích các PNG, JSON bên dưới `nerf_synthetic/lego/`. ZIP có 370.385.516 byte. Dữ liệu Lego sau giải nén có 250.888.858 byte. [Script tải dữ liệu của tác giả](https://github.com/bmild/nerf/blob/master/download_example_data.sh)

Chạy từ gốc dự án mới, sau đó chạy các lệnh kiểm tra ở trên. Tùy chọn `-n` của `unzip` bảo vệ các tệp đã có.

```sh
(
  set -eu
  staging_dir=$(mktemp -d)
  archive="$staging_dir/nerf_example_data.zip"
  curl -fL --retry 3 -o "$archive" 'https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip'
  printf '%s  %s\n' 'ce4e94e031c099a19ef04cfb6c71f1e47225d97d365be610b476e379a386c25f' "$archive" | shasum -a 256 -c -
  mkdir -p data/raw
  unzip -nq "$archive" 'nerf_synthetic/lego/*.png' 'nerf_synthetic/lego/*.json' -d data/raw
)
```
