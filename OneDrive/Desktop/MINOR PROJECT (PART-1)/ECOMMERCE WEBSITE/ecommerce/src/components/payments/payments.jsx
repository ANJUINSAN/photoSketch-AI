import React from "react";

import './payments.css'
import 'mdb-react-ui-kit/dist/css/mdb.min.css';
import { ShopContext } from "../../context/shopcontext";
import { useContext } from "react";


export default function Payment() {
  const { getTotalCartAmount } =
    useContext(ShopContext);

  return (
   <>
   <section>
  <div className="container py-5">
    <div className="row d-flex justify-content-center">
      <div className="col-md-12 col-lg-10 col-xl-8">
        <div className="card">
          <div className="card-body p-md-5">
            <div>
              <h4>Upgrade your plan</h4>
              <p className="text-muted pb-2">
                Please make the payment to start enjoying all the features of our premium
                plan as soon as possible
              </p>
            </div>

            <div className="px-3 py-4 border border-primary border-2 rounded mt-4 d-flex justify-content-between">
              <div className="d-flex flex-row align-items-center">
                <img src="https://i.imgur.com/S17BrTx.webp" className="rounded" width="60" />
                <div className="d-flex flex-column ms-4">
                  <span className="h5 mb-1">Small Business</span>
                  <span className="small text-muted">CHANGE PLAN</span>
                </div>
              </div>
              <div className="d-flex flex-row align-items-center">
                <sup className="dollar font-weight-bold text-muted">$</sup>
                <span className="h2 mx-1 mb-0">${getTotalCartAmount()}</span>
                <span className="text-muted font-weight-bold mt-2">/ year</span>
              </div>
            </div>

            <h4 className="mt-5">Payment details</h4>

            <div className="mt-4 d-flex justify-content-between align-items-center">
              <div className="d-flex flex-row align-items-center">
                <img src="https://i.imgur.com/qHX7vY1.webp" className="rounded" width="70" />
                <div className="d-flex flex-column ms-3">
                  <span className="h5 mb-1">Credit Card</span>
                  <span className="small text-muted">1234 XXXX XXXX 2570</span>
                </div>
              </div>
              <div>
                <input type="text" className="form-control" placeholder="CVC" />
              </div>
            </div>

            <div className="mt-2 d-flex justify-content-between align-items-center">
              <div className="d-flex flex-row align-items-center">
                <img src="https://i.imgur.com/qHX7vY1.webp" className="rounded" width="70" />
                <div className="d-flex flex-column ms-3">
                  <span className="h5 mb-1">Credit Card</span>
                  <span className="small text-muted">2344 XXXX XXXX 8880</span>
                </div>
              </div>
              <div>
                <input type="text" className="form-control" placeholder="CVC" />
              </div>
            </div>

            <h6 className="mt-4 mb-3 text-primary text-uppercase">ADD PAYMENT METHOD</h6>

            <div className="form-outline">
              <input type="text" id="formControlLg" className="form-control form-control-lg" />
              <label className="form-label" htmlFor="formControlLg">Email address</label>
            </div>
            <div className="mt-3">
              <button className="btn btn-primary btn-block btn-lg">
                Proceed to payment <i className="fas fa-long-arrow-alt-right"></i>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>
   </>
  );
}