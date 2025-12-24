# __all__ = ['train_epoch', 'test_epoch']
#
# import torch
#
#
# def train_epoch(model, train_loader, loss, optimizer, device):
#     model.train()
#
#     total_loss = 0
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#
#         output = model(data)
#
#         loss_val = loss(output, target)
#         loss_val.backward()
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#         total_loss += loss_val.item()
#     return total_loss / len(train_loader)
#
#
# def test_epoch(model, tets_loader, loss, device):
#     model.eval()
#
#     total_loss = 0
#     with torch.no_grad():
#         for data, target in tets_loader:
#             data, target = data.to(device), target.to(device)
#
#             output = model(data)
#
#             loss_val = loss(output, target)
#             total_loss += loss_val.item()
#
#     return total_loss / len(tets_loader)
